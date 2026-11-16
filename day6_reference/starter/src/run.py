"""Run the Day 06 ingestion and retrieval benchmark pipeline.

Run from starter/:
    python -m src.run

The pipeline uses the repository manifest and benchmark, one real local
Sentence Transformers model, and PostgreSQL + pgvector. It never substitutes
mock vectors when model/database setup fails.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version as package_version
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import chunk_page, chunk_statistics
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf


LOGGER = logging.getLogger("day6.pipeline")

STRATEGIES = ("A", "B", "C")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
MIN_CONTENT_PAGES = 30
COHERENCE_SAMPLE_SIZE = 10
GROUNDING_SAMPLE_SIZE = 5
RANDOM_SEED = 42

SECTION_TITLES = {
    1: "Chapter 1 - Ethics of Research Abroad",
    2: "Chapter 2 - Project Development with International Partners",
    3: "Chapter 3 - Guidelines for Professional Behavior Abroad",
    4: "Chapter 4 - Global Citizenship and Advocacy",
    5: "Chapter 5 - Logistics of Research and Service Abroad",
}



def _package_versions() -> dict[str, str | None]:
    packages = {
        "PyMuPDF": "PyMuPDF",
        "sentence-transformers": "sentence-transformers",
        "psycopg": "psycopg",
        "pgvector": "pgvector",
        "python-dotenv": "python-dotenv",
    }
    versions: dict[str, str | None] = {}
    for label, distribution in packages.items():
        try:
            versions[label] = package_version(distribution)
        except PackageNotFoundError:
            versions[label] = None
    return versions


def _project_root() -> Path:
    # run.py -> src -> starter -> repository root
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _normalize_for_evidence(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _page_header(text: str) -> str:
    """Return the first non-empty lines, used only to identify source headings."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:600]


def _attach_source_metadata(
    pages: list[Any],
    manifest: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Select useful pages and attach section metadata grounded in page headings.

    The handbook contains front matter and a license page. The corpus keeps
    the substantive Introduction page and pages whose running header identifies
    one of the five handbook chapters. This excludes cover, contributors/TOC,
    license and post-chapter policy/contact pages.
    """
    manifest_pages = {
        int(item["page_number"]): item
        for item in manifest.get("pages", [])
        if "page_number" in item
    }

    selected = []
    excluded = []

    for page in pages:
        manifest_item = manifest_pages.get(page.page, {})
        if manifest_item and not manifest_item.get("include_in_corpus", False):
            excluded.append({
                "page": page.page,
                "reason": "manifest_include_in_corpus_false",
            })
            continue

        header = _page_header(page.text)
        section = None

        # The first pages contain contributors and a table of contents that
        # mention chapter names but are not chapter content. Keep the
        # substantive introduction page separately; begin chapter matching
        # at physical PDF page 8.
        is_introduction = (
            "traveling abroad for research or service" in page.text.casefold()
            and "five major areas" in page.text.casefold()
        )
        if page.page < 8 and not is_introduction:
            excluded.append({
                "page": page.page,
                "reason": "front_matter_or_license_page",
            })
            continue

        # Chapter names are copied from the actual handbook chapter headings.
        for chapter_number, canonical_value in SECTION_TITLES.items():
            title = canonical_value.split(" - ", 1)[1]
            if title.casefold() in header.casefold():
                section = canonical_value
                break

        # The handbook introduction is substantive content and contains Q001's
        # evidence. It is distinct from the license and table-of-contents pages.

        if section is None and is_introduction:
            section = "Introduction"

        if section is None:
            excluded.append({
                "page": page.page,
                "reason": "not_a_substantive_chapter_or_introduction_page",
            })
            continue

        page.metadata = dict(page.metadata or {})
        page.metadata.update({
            "page_id": manifest_item.get(
                "page_id",
                f"{page.source}-p{page.page:03d}",
            ),
            "section": section,
            "section_source": "heading_in_source_page",
        })
        selected.append(page)

    selected.sort(key=lambda item: item.page)
    return selected, excluded


def _validate_corpus(pages: list[Any]) -> None:
    content_pages = [
        page for page in pages
        if isinstance(page.text, str) and page.text.strip()
    ]
    if len(content_pages) < MIN_CONTENT_PAGES:
        raise ValueError(
            f"Corpus contains {len(content_pages)} useful content pages; "
            f"the GUIDE requires at least {MIN_CONTENT_PAGES}. "
            "Review the manifest and page-selection rules."
        )


def _normalize_filter(question: dict[str, Any]) -> dict[str, Any] | None:
    """Convert benchmark {field, value} to the vector-store filter mapping."""
    raw = question.get("retrieval_filter", question.get("filter"))
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"{question.get('question_id')}: filter must be an object or null."
        )

    # Current benchmark format: {"field": "section", "value": "..."}.
    if "field" in raw:
        field = raw.get("field")
        value = raw.get("value")
        if not field or value is None:
            raise ValueError(
                f"{question.get('question_id')}: filter needs field and value."
            )
        return {str(field): value}

    # Also accept a direct mapping, e.g. {"section": "..."}.
    return raw


def _validate_benchmark(
    benchmark: dict[str, Any],
    corpus_pages: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        raise ValueError("benchmark/questions.json must contain a questions list.")

    if len(questions) < 20:
        raise ValueError(f"Benchmark has {len(questions)} questions; at least 20 required.")

    split_counts = {
        split: sum(q.get("split") == split for q in questions)
        for split in ("dev", "test")
    }
    if split_counts["dev"] < 15 or split_counts["test"] < 5:
        raise ValueError(
            f"Benchmark split must contain at least 15 dev / 5 test; got {split_counts}."
        )

    ids = [q.get("question_id") for q in questions]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("Benchmark question_id values must be unique and non-empty.")

    allowed_filter_values = (
        benchmark.get("metadata_filter", {}).get("allowed_values", [])
    )
    filter_field = benchmark.get("metadata_filter", {}).get("field")

    filter_questions = [
        q for q in questions
        if _normalize_filter(q) is not None
    ]
    for question in filter_questions:
        normalized_filter = _normalize_filter(question)
        if filter_field and filter_field not in normalized_filter:
            raise ValueError(
                f"{question['question_id']}: filter must use metadata field "
                f"{filter_field!r}."
            )
        if allowed_filter_values:
            for field, value in normalized_filter.items():
                if field == filter_field and value not in allowed_filter_values:
                    raise ValueError(
                        f"{question['question_id']}: filter value {value!r} "
                        "is not listed in benchmark metadata_filter.allowed_values."
                    )
    filter_test_questions = [
        q for q in filter_questions if q.get("split") == "test"
    ]
    if len(filter_questions) < 3 or not filter_test_questions:
        raise ValueError(
            "Benchmark requires at least 3 filtered questions, including at least 1 test question."
        )

    page_texts = {
        (page.source, page.page): page.text
        for page in corpus_pages
    }
    validation = []

    for question in questions:
        if question.get("split") not in {"dev", "test"}:
            raise ValueError(f"{question.get('question_id')}: split must be dev or test.")
        if not str(question.get("question", "")).strip():
            raise ValueError(f"{question.get('question_id')}: question cannot be empty.")
        if not str(question.get("expected_answer", "")).strip():
            raise ValueError(f"{question.get('question_id')}: expected_answer is required.")

        evidence_items = question.get("gold_evidence", [])
        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError(
                f"{question.get('question_id')}: at least one gold_evidence item is required."
            )

        for evidence in evidence_items:
            source = evidence.get("source")
            page_number = evidence.get("page")
            gold_text = evidence.get("text", "")
            source_page_text = page_texts.get((source, page_number))

            if source_page_text is None:
                validation.append({
                    "question_id": question["question_id"],
                    "status": "invalid_source_or_page",
                    "source": source,
                    "page": page_number,
                    "message": "Gold source/page is not present in the selected corpus.",
                })
                continue

            normalized_gold = _normalize_for_evidence(gold_text)
            normalized_source = _normalize_for_evidence(source_page_text)
            exact = bool(normalized_gold and normalized_gold in normalized_source)

            validation.append({
                "question_id": question["question_id"],
                "status": "verified_exact" if exact else "review_required",
                "source": source,
                "page": page_number,
                "message": (
                    "Gold evidence found on the declared physical PDF page."
                    if exact else
                    "Evidence is not a contiguous exact match after normalization; "
                    "review whether it combines separate passages or has an incorrect quote/page."
                ),
                "evidence_sufficiency": "manual_review_required",
                "sufficiency_check": (
                    "Confirm that the evidence supports every part of expected_answer, "
                    "including conditions, exceptions and all requested items."
                ),
            })

    invalid = [item for item in validation if item["status"] == "invalid_source_or_page"]
    if invalid:
        raise ValueError(
            "Benchmark contains gold source/page references outside the selected corpus: "
            + json.dumps(invalid, ensure_ascii=False)
        )

    # Non-contiguous evidence is surfaced as a review warning, not silently
    # rewritten. The benchmark owner must verify that it is sufficient.
    return questions, validation


def _build_all_chunks(pages: list[Any]) -> tuple[dict[str, list[Any]], dict[str, float]]:
    chunks_by_strategy: dict[str, list[Any]] = {}
    timings: dict[str, float] = {}

    for strategy in STRATEGIES:
        started = time.perf_counter()
        chunks = []
        for page in pages:
            chunks.extend(
                chunk_page(
                    page,
                    strategy=strategy,
                    size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )
            )
        timings[strategy] = time.perf_counter() - started
        if not chunks:
            raise ValueError(f"Strategy {strategy} produced no chunks.")
        chunks_by_strategy[strategy] = chunks

    # B and C must have identical chunk text and boundaries.
    b_signature = [
        (c.document_id, c.source, c.page, c.text)
        for c in chunks_by_strategy["B"]
    ]
    c_signature = [
        (c.document_id, c.source, c.page, c.text)
        for c in chunks_by_strategy["C"]
    ]
    if b_signature != c_signature:
        raise RuntimeError("B and C chunk boundaries differ; only filtering may differ.")

    return chunks_by_strategy, timings


def _metadata_check(
    hit: dict[str, Any],
    filters: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = hit.get("metadata") or {}
    has_trace_fields = bool(
        metadata.get("source")
        and metadata.get("page") is not None
        and metadata.get("document_id")
    )
    filter_match = (
        all(metadata.get(key) == value for key, value in filters.items())
        if filters else None
    )
    return {
        "source_present": bool(metadata.get("source")),
        "page_present": metadata.get("page") is not None,
        "document_id_present": bool(metadata.get("document_id")),
        "trace_fields_present": has_trace_fields,
        "filter_required": bool(filters),
        "filter_match": filter_match,
        "metadata": metadata,
    }


def _prepare_coherence_assessment(
    pages: list[Any],
    chunks_by_strategy: dict[str, list[Any]],
) -> dict[str, Any]:
    """Use the same first ten sorted source pages for all strategies."""
    selected_pages = sorted(pages, key=lambda page: page.page)[:COHERENCE_SAMPLE_SIZE]
    samples = []

    for page in selected_pages:
        for strategy in STRATEGIES:
            page_chunks = [
                chunk for chunk in chunks_by_strategy[strategy]
                if chunk.page == page.page and chunk.source == page.source
            ]
            samples.append({
                "sample_id": f"page-{page.page:03d}-{strategy}",
                "source_position": {
                    "source": page.source,
                    "page": page.page,
                    "page_id": page.metadata.get("page_id"),
                },
                "strategy": strategy,
                "source_text": page.text,
                "chunks_covering_source_position": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "manual_score_0_to_2": None,
                        "reason": "",
                    }
                    for chunk in page_chunks
                ],
                "manual_sample_score_0_to_2": None,
                "sample_reason": "",
            })

    return {
        "method": (
            "Same first 10 sorted content-page positions are assessed for A, B and C, "
            "following GUIDE.md. Review all chunks covering each source page."
        ),
        "score_scale": {
            "0": "Chunk is cut mid-idea or lacks the subject needed to understand it.",
            "1": "Partly understandable but missing context.",
            "2": "Self-contained and preserves the complete idea.",
        },
        "assessor": "To be completed by the student; record assessor identity and bias risk.",
        "samples": samples,
        "manual_summary": {
            "A_mean_out_of_2": None,
            "B_mean_out_of_2": None,
            "C_mean_out_of_2": None,
        },
    }


def _prepare_manual_audits(
    traces: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    test_questions = [q for q in questions if q.get("split") == "test"][:GROUNDING_SAMPLE_SIZE]
    test_ids = [q["question_id"] for q in test_questions]

    metadata_samples = []
    grounding_samples = []

    for question in test_questions:
        for strategy in STRATEGIES:
            trace = next(
                (
                    item for item in traces
                    if item["question_id"] == question["question_id"]
                    and item["strategy"] == strategy
                ),
                None,
            )
            if trace is None:
                raise RuntimeError(
                    f"Missing trace for {question['question_id']} / {strategy}."
                )

            filters = trace["applied_filter"]
            metadata_samples.append({
                "question_id": question["question_id"],
                "question": question["question"],
                "strategy": strategy,
                "applied_filter": filters,
                "top3_metadata": [
                    {
                        "rank": rank,
                        "chunk_id": hit.get("chunk_id"),
                        **_metadata_check(hit, filters),
                        "text_preview": (hit.get("text") or "")[:300],
                        "manual_source_correct": None,
                        "manual_page_correct": None,
                        "manual_filter_value_correct": None,
                        "manual_reason": "",
                    }
                    for rank, hit in enumerate(trace["hits"], start=1)
                ],
                "manual_assessment": {
                    "source_correct": None,
                    "page_correct": None,
                    "filter_field_correct": None,
                    "reason": "",
                },
            })

            grounding_samples.append({
                "question_id": question["question_id"],
                "question": question["question"],
                "expected_answer_for_context_only": question.get("expected_answer"),
                "strategy": strategy,
                "gold_evidence": question.get("gold_evidence", []),
                "top3_evidence": [
                    {
                        "rank": rank,
                        "chunk_id": hit.get("chunk_id"),
                        "source": (hit.get("metadata") or {}).get("source"),
                        "page": (hit.get("metadata") or {}).get("page"),
                        "score": hit.get("score"),
                        "text": hit.get("text"),
                    }
                    for rank, hit in enumerate(trace["hits"], start=1)
                ],
                "manual_score_0_to_2": None,
                "reason": "",
            })

    return (
        {
            "method": "Manual audit on the same five test questions for each strategy, top-3 per question.",
            "questions": test_ids,
            "score_scale": "Check source/page and required metadata against the source document.",
            "samples": metadata_samples,
            "summary_to_complete": {
                "valid_metadata_records": None,
                "total_records_checked": sum(len(s["top3_metadata"]) for s in metadata_samples),
                "filter_questions": [
                    q["question_id"] for q in test_questions
                    if _normalize_filter(q) is not None
                ],
                "C_filter_compliant_questions": None,
                "C_filter_questions_checked": None,
                "useful_metadata_field_and_reason": "",
                "metadata_limitation": "",
            },
        },
        {
            "method": "Manual assessment of retrieved evidence only; this is not LLM answer accuracy.",
            "questions": test_ids,
            "score_scale": {
                "0": "No supporting evidence in top-3.",
                "1": "Evidence exists but misses a required condition or context.",
                "2": "Evidence is sufficient to answer and can be traced to its source.",
            },
            "samples": grounding_samples,
            "summary_to_complete": {
                "A_score_counts_0_1_2": None,
                "A_mean_out_of_2": None,
                "B_score_counts_0_1_2": None,
                "B_mean_out_of_2": None,
                "C_score_counts_0_1_2": None,
                "C_mean_out_of_2": None,
                "assessor": "",
                "bias_risk": "",
            },
        },
    )


def _metric_rate(
    summary: dict[str, Any],
    strategy: str,
    split: str,
    metric: str,
) -> float | None:
    return (
        summary.get(strategy, {})
        .get(split, {})
        .get(metric, {})
        .get("rate")
    )


def _prepare_strategy_impact(
    chunks_by_strategy: dict[str, list[Any]],
    chunk_times: dict[str, float],
    resource_stats: dict[str, Any],
    retrieval_times: dict[str, list[float]],
    retrieval_summary: dict[str, Any],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons = {}

    for left, right in (("A", "B"), ("B", "C")):
        comparisons[f"{left}_vs_{right}"] = {
            "chunk_count_delta_right_minus_left": (
                len(chunks_by_strategy[right]) - len(chunks_by_strategy[left])
            ),
            "dev_hit_at_1_delta": (
                (_metric_rate(retrieval_summary, right, "dev", "hit_at_1") or 0)
                - (_metric_rate(retrieval_summary, left, "dev", "hit_at_1") or 0)
            ),
            "dev_evidence_hit_at_3_delta": (
                (_metric_rate(retrieval_summary, right, "dev", "evidence_hit_at_3") or 0)
                - (_metric_rate(retrieval_summary, left, "dev", "evidence_hit_at_3") or 0)
            ),
            "test_hit_at_1_delta": (
                (_metric_rate(retrieval_summary, right, "test", "hit_at_1") or 0)
                - (_metric_rate(retrieval_summary, left, "test", "hit_at_1") or 0)
            ),
            "test_evidence_hit_at_3_delta": (
                (_metric_rate(retrieval_summary, right, "test", "evidence_hit_at_3") or 0)
                - (_metric_rate(retrieval_summary, left, "test", "evidence_hit_at_3") or 0)
            ),
            "interpretation_to_complete": "",
        }

    failures = [
        {
            "question_id": trace["question_id"],
            "split": trace["split"],
            "strategy": trace["strategy"],
            "top3_chunk_ids": [hit.get("chunk_id") for hit in trace["hits"]],
            "note": "No automatic full-evidence match in top-3; inspect whether evidence was split or retrieval missed it.",
        }
        for trace in traces
        if not trace["evaluation"]["evidence_hit_at_3"]
    ]

    return {
        "resource_stats": resource_stats,
        "pairwise_comparisons": comparisons,
        "retrieval_time_seconds_by_strategy": {
            strategy: {
                "total": round(sum(values), 6),
                "mean": round(sum(values) / len(values), 6) if values else None,
                "query_count": len(values),
            }
            for strategy, values in retrieval_times.items()
        },
        "failure_examples": failures[:30],
        "tradeoffs_and_final_choice_to_complete": {
            "A_vs_B_chunking_tradeoff": "",
            "B_vs_C_metadata_filter_tradeoff": "",
            "failure_analysis": "",
            "selection_rule_fixed_before_test_review": "",
            "chosen_strategy_and_reason": "",
            "limitations_small_test_set_and_corpus_scope": "",
        },
    }


def main() -> None:
    """Execute ingestion, three strategy indexes, retrieval and assessments."""
    started_all = time.perf_counter()

    project_root = _project_root()
    starter_dir = project_root / "starter"
    load_dotenv(starter_dir / ".env")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest_path = Path(os.getenv(
        "MANIFEST_PATH",
        str(project_root / "data" / "manifest.json"),
    ))
    benchmark_path = Path(os.getenv(
        "BENCHMARK_PATH",
        str(project_root / "benchmark" / "questions.json"),
    ))
    results_dir = Path(os.getenv(
        "RESULTS_DIR",
        str(starter_dir / "results"),
    ))
    database_url = os.getenv("DATABASE_URL", "").strip()
    model_name = os.getenv("EMBEDDING_MODEL", "").strip() or MyEmbedder.DEFAULT_MODEL

    if not database_url:
        raise ValueError(
            "DATABASE_URL is empty. Copy starter/.env.example to starter/.env "
            "and configure PostgreSQL + pgvector."
        )

    manifest = _load_json(manifest_path)
    benchmark = _load_json(benchmark_path)

    source_id = manifest.get("source_id")
    if not source_id:
        raise ValueError("manifest.json must define source_id.")

    source_path = Path(manifest.get("source_path_or_url", ""))
    if not source_path.is_absolute():
        source_path = project_root / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Corpus PDF not found: {source_path}")

    benchmark_source = benchmark.get("source_id")
    if benchmark_source and benchmark_source != source_id:
        raise ValueError(
            f"Benchmark source_id {benchmark_source!r} does not match "
            f"manifest source_id {source_id!r}."
        )

    print("=" * 72)
    print("DAY 06 - DATA INGESTION & RETRIEVAL BENCHMARK")
    print("=" * 72)

    # 1. Ingest and select content pages.
    ingest_started = time.perf_counter()
    document_id = benchmark.get("benchmark_id") or source_id
    all_pages = load_pdf(
        source_path,
        document_id=document_id,
        source=source_id,
    )
    pages, excluded_pages = _attach_source_metadata(all_pages, manifest)
    _validate_corpus(pages)
    ingest_seconds = time.perf_counter() - ingest_started

    print(
        f"Corpus: {len(pages)} content pages selected from "
        f"{len(all_pages)} physical PDF pages."
    )

    questions, gold_validation = _validate_benchmark(benchmark, pages)
    review_items = [
        item for item in gold_validation
        if item["status"] != "verified_exact"
    ]
    if review_items:
        LOGGER.warning(
            "Some gold evidence needs manual review; see gold_validation.json: %s",
            json.dumps(review_items, ensure_ascii=False),
        )

    # 2. Build A/B/C chunks. C is checked against B for exact boundary parity.
    chunks_by_strategy, chunking_times = _build_all_chunks(pages)

    # 3. Load one real model and use it for every strategy and query.
    embedder = MyEmbedder(model_name=model_name)
    try:
        from .store import PgVectorStore
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL dependencies are missing. Install starter/requirements.txt."
        ) from exc

    store = PgVectorStore(
        database_url=database_url,
        dimensions=embedder.dimensions,
    )
    store.create_schema()

    resource_stats: dict[str, Any] = {}
    for strategy in STRATEGIES:
        chunks = chunks_by_strategy[strategy]

        embedding_started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        embedding_seconds = time.perf_counter() - embedding_started

        indexing_started = time.perf_counter()
        store.upsert(strategy=strategy, chunks=chunks, vectors=vectors)
        indexing_seconds = time.perf_counter() - indexing_started

        resource_stats[strategy] = {
            "chunking": chunk_statistics(chunks),
            "chunking_time_seconds": round(chunking_times[strategy], 4),
            "embedding_time_seconds": round(embedding_seconds, 4),
            "indexing_time_seconds": round(indexing_seconds, 4),
            "embedding_model": embedder.model_name,
            "embedding_dimensions": embedder.dimensions,
            "embedding_provider": "local_sentence_transformers",
            "embedding_api_cost": 0,
            "vector_store": "PostgreSQL + pgvector",
            "similarity_metric": "cosine",
            "vector_index": "HNSW configured; actual planner path depends on PostgreSQL query plan",
        }

    # 4. Same question vector for A/B/C; only C receives a question filter.
    traces: list[dict[str, Any]] = []
    retrieval_times: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}

    for question in questions:
        query_vector = embedder.embed_query(question["question"])
        question_filter = _normalize_filter(question)

        for strategy in STRATEGIES:
            applied_filter = question_filter if strategy == "C" else None

            retrieval_started = time.perf_counter()
            hits = store.search(
                strategy=strategy,
                query_vector=query_vector,
                top_k=TOP_K,
                filters=applied_filter,
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            retrieval_times[strategy].append(retrieval_seconds)

            evaluation_question = {
                **question,
                "strategy": strategy,
            }
            evaluation = evaluate_query(evaluation_question, hits)

            traces.append({
                "question_id": question["question_id"],
                "question": question["question"],
                "split": question["split"],
                "type": question.get("type"),
                "strategy": strategy,
                "applied_filter": applied_filter,
                "gold_evidence": question.get("gold_evidence", []),
                "alternative_evidence": question.get("alternative_evidence", []),
                "hits": hits[:TOP_K],
                "evaluation": evaluation,
                "retrieval_time_seconds": round(retrieval_seconds, 6),
            })

    retrieval_summary = summarize([
        trace["evaluation"] for trace in traces
    ])

    # 5. Build all manual assessment artifacts required by GUIDE.md.
    coherence_assessment = _prepare_coherence_assessment(
        pages,
        chunks_by_strategy,
    )
    metadata_assessment, grounding_assessment = _prepare_manual_audits(
        traces,
        questions,
    )
    strategy_impact = _prepare_strategy_impact(
        chunks_by_strategy=chunks_by_strategy,
        chunk_times=chunking_times,
        resource_stats=resource_stats,
        retrieval_times=retrieval_times,
        retrieval_summary=retrieval_summary,
        traces=traces,
    )

    # Metadata auto-audit for the five test questions, top-3 each.
    metadata_auto_summary = {}
    for strategy in STRATEGIES:
        strategy_records = [
            sample for sample in metadata_assessment["samples"]
            if sample["strategy"] == strategy
        ]
        hits = [
            hit for sample in strategy_records
            for hit in sample["top3_metadata"]
        ]
        valid = sum(hit["trace_fields_present"] for hit in hits)
        filtered_samples = [
            sample for sample in strategy_records
            if sample["applied_filter"] is not None
        ] if strategy == "C" else []
        compliant_questions = sum(
            bool(sample["top3_metadata"])
            and all(hit["filter_match"] is True for hit in sample["top3_metadata"])
            for sample in filtered_samples
        )
        filter_records = [
            hit for sample in filtered_samples
            for hit in sample["top3_metadata"]
        ]
        filter_valid = sum(hit["filter_match"] is True for hit in filter_records)
        metadata_auto_summary[strategy] = {
            "trace_fields_present": valid,
            "retrieved_records_checked": len(hits),
            "trace_field_presence_rate": valid / len(hits) if hits else None,
            "filter_compliant_questions": compliant_questions,
            "filter_questions_checked": len(filtered_samples),
            "filter_compliance_rate_by_question": (
                compliant_questions / len(filtered_samples)
                if filtered_samples else None
            ),
            "filter_compliant_records": filter_valid,
            "filter_records_checked": len(filter_records),
            "filter_compliance_rate_by_record": (
                filter_valid / len(filter_records) if filter_records else None
            ),
            "note": "Presence/filter are checked automatically; source/page correctness still requires manual verification against the PDF.",
        }
    metadata_assessment["automatic_checks"] = metadata_auto_summary
    metadata_assessment["summary_to_complete"]["C_filter_compliant_questions"] = (
        metadata_auto_summary["C"]["filter_compliant_questions"]
    )
    metadata_assessment["summary_to_complete"]["C_filter_questions_checked"] = (
        metadata_auto_summary["C"]["filter_questions_checked"]
    )
    metadata_assessment["summary_to_complete"]["C_filter_compliance_rate"] = (
        metadata_auto_summary["C"]["filter_compliance_rate_by_question"]
    )

    # Persist reproducibility and complete traces.
    run_seconds = time.perf_counter() - started_all
    _save_json(results_dir / "retrieval_traces.json", traces)
    for split in ("dev", "test"):
        _save_json(
            results_dir / f"{split}_traces.json",
            [trace for trace in traces if trace["split"] == split],
        )

    _save_json(results_dir / "retrieval_metrics.json", retrieval_summary)
    _save_json(results_dir / "coherence_assessment.json", coherence_assessment)
    _save_json(results_dir / "metadata_assessment.json", metadata_assessment)
    _save_json(results_dir / "grounding_assessment.json", grounding_assessment)
    _save_json(results_dir / "strategy_impact.json", strategy_impact)
    _save_json(results_dir / "gold_validation.json", {
        "source_id": source_id,
        "manifest_path": str(manifest_path),
        "benchmark_path": str(benchmark_path),
        "results": gold_validation,
        "review_required": review_items,
    })

    run_summary = {
        "source_id": source_id,
        "document_id": document_id,
        "source_title": manifest.get("title"),
        "source_file": str(source_path),
        "source_license_or_permission": manifest.get("license_or_permission"),
        "source_accessed_at": manifest.get("accessed_at"),
        "source_version": manifest.get("file_metadata", {}).get("version"),
        "source_sha256": manifest.get("file_metadata", {}).get("file_sha256"),
        "content_page_count": len(pages),
        "content_page_numbers": [page.page for page in pages],
        "empty_physical_pages": [
            page.page for page in all_pages if not page.text.strip()
        ],
        "corpus_total_characters": sum(len(page.text) for page in pages),
        "excluded_pages": excluded_pages,
        "benchmark_question_count": len(questions),
        "dev_question_count": sum(q["split"] == "dev" for q in questions),
        "test_question_count": sum(q["split"] == "test" for q in questions),
        "filtered_question_count": sum(
            _normalize_filter(q) is not None for q in questions
        ),
        "strategies": list(STRATEGIES),
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector",
        "similarity_metric": "cosine",
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "ingestion_time_seconds": round(ingest_seconds, 4),
        "total_pipeline_time_seconds": round(run_seconds, 4),
        "manual_assessment_status": "templates_created_not_yet_scored",
        "outputs": [
            "retrieval_traces.json",
            "dev_traces.json",
            "test_traces.json",
            "retrieval_metrics.json",
            "coherence_assessment.json",
            "metadata_assessment.json",
            "grounding_assessment.json",
            "strategy_impact.json",
            "gold_validation.json",
            "run_summary.json",
        ],
        "reproducibility_notes": [
            "Same corpus, benchmark questions, embedding model and cosine metric used for A/B/C.",
            "B and C have identical chunk text and boundaries; C applies metadata filters only when a benchmark filter is present.",
            "No mock embeddings or silent fallback are used.",
            "Re-running replaces chunks for each strategy/document instead of appending duplicate rows.",
            "Test results must not be used to tune the strategy.",
        ],
    }
    _save_json(results_dir / "run_summary.json", run_summary)

    print("\nRetrieval summary (Hit@1 / Evidence Hit@3):")
    print(json.dumps(retrieval_summary, ensure_ascii=False, indent=2))
    print(f"\nCompleted in {run_seconds:.2f}s. Results: {results_dir}")


if __name__ == "__main__":
    main()
