
"""Evaluate retrieved evidence and preserve per-query traces.

Expected benchmark question format:
{
    "question_id": "Q01",
    "question": "What is ...?",
    "split": "dev",  # or "test"
    "strategy": "A",
    "gold_evidence": [
        {
            "source": "student_handbook_for_global_engagement",
            "page": 5,
            "text": "The exact evidence text from the source."
        }
    ],
    "alternative_evidence": [
        {
            "source": "another_document_id",
            "page": 3,
            "text": "Another valid evidence passage."
        }
    ]
}

Expected hit format:
{
    "chunk_id": "...",
    "text": "...",
    "score": 0.85,
    "metadata": {
        "source": "student_handbook_for_global_engagement",
        "page": 5
    }
}

Evidence matching uses normalized text containment. It does not
automatically assign human grounding or chunk coherence labels.
"""

from typing import Any
from collections import defaultdict
import re
import unicodedata


def _normalize_text(text: Any) -> str:
    """Normalize text for a basic evidence containment check."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _get_hit_metadata(hit: dict[str, Any]) -> dict[str, Any]:
    """Support metadata stored either inside or outside a metadata object."""
    metadata = hit.get("metadata")

    if isinstance(metadata, dict):
        return metadata

    return hit


def _get_gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect primary and documented alternative gold evidence."""
    evidence_items = []

    primary = question.get("gold_evidence", [])
    alternatives = question.get("alternative_evidence", [])

    # Support a single evidence object as well as a list of objects.
    if isinstance(primary, dict):
        primary = [primary]
    if isinstance(alternatives, dict):
        alternatives = [alternatives]

    if isinstance(primary, list):
        evidence_items.extend(primary)

    if isinstance(alternatives, list):
        evidence_items.extend(alternatives)

    # Support a simpler benchmark format using gold_source/gold_page.
    if not evidence_items and question.get("gold_source") is not None:
        evidence_items.append({
            "source": question.get("gold_source"),
            "page": question.get("gold_page"),
            "text": question.get("gold_evidence_text", "")
        })

    return [
        item for item in evidence_items
        if isinstance(item, dict)
    ]


def _same_source_and_page(
    hit_metadata: dict[str, Any],
    gold: dict[str, Any]
) -> bool:
    """Check whether a retrieved chunk belongs to the gold source and page."""
    hit_source = hit_metadata.get("source")
    gold_source = gold.get("source")

    hit_page = hit_metadata.get("page")
    gold_page = gold.get("page")

    if hit_source is None or gold_source is None:
        return False

    if str(hit_source).strip() != str(gold_source).strip():
        return False

    # If page is specified in gold evidence, it must match.
    if gold_page is not None:
        if hit_page is None or str(hit_page) != str(gold_page):
            return False

    return True


def _contains_evidence(
    hit_text: str,
    gold_text: str
) -> bool:
    """Check whether the retrieved chunk contains the gold evidence text."""
    normalized_hit = _normalize_text(hit_text)
    normalized_gold = _normalize_text(gold_text)

    if not normalized_hit or not normalized_gold:
        return False

    return normalized_gold in normalized_hit


def _matches_gold(
    hit: dict[str, Any],
    gold_evidence: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any] | None]:
    """Return whether a hit matches any valid gold evidence."""
    metadata = _get_hit_metadata(hit)
    hit_text = hit.get("text", "")

    for gold in gold_evidence:
        if not _same_source_and_page(metadata, gold):
            continue

        gold_text = gold.get("text", gold.get("evidence_text", ""))

        if _contains_evidence(hit_text, gold_text):
            return True, gold

    return False, None


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate Hit@1 and Evidence Hit@3 and preserve the retrieval trace.

    A hit is considered relevant only when:
    1. Its source matches a documented gold source.
    2. Its page matches when a gold page is specified.
    3. Its text contains the documented gold evidence passage.

    Alternative gold evidence is accepted.

    This is a text-based evidence check. Human-confirmed equivalent
    evidence should be documented separately rather than inferred here.
    """
    if not isinstance(question, dict):
        raise TypeError("question must be a dictionary.")

    if not isinstance(hits, list):
        raise TypeError("hits must be a list.")

    gold_evidence = _get_gold_evidence(question)

    if not gold_evidence:
        raise ValueError(
            f"Question {question.get('question_id', 'unknown')} "
            "has no valid gold evidence."
        )

    # Only the first three retrieved results are used for evaluation.
    top_hits = hits[:3]

    hit_traces = []

    for rank, hit in enumerate(top_hits, start=1):
        if not isinstance(hit, dict):
            raise TypeError("Each retrieved hit must be a dictionary.")

        matched, matched_gold = _matches_gold(hit, gold_evidence)
        metadata = _get_hit_metadata(hit)

        hit_traces.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "text": hit.get("text", ""),
            "score": hit.get("score", hit.get("similarity")),
            "metadata": metadata,
            "evidence_match": matched,
            "matched_gold_evidence": matched_gold
        })

    hit_at_1 = (
        hit_traces[0]["evidence_match"]
        if hit_traces else False
    )

    evidence_hit_at_3 = any(
        item["evidence_match"] for item in hit_traces
    )

    return {
        "question_id": question.get("question_id"),
        "question": question.get("question"),
        "split": question.get("split", "unknown"),
        "strategy": question.get("strategy", "unknown"),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "retrieved_count": len(top_hits),
        "gold_evidence": gold_evidence,
        "hits": hit_traces
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split.

    Each question is counted once in the denominator, including
    questions for which no relevant evidence was retrieved.
    """
    if not isinstance(traces, list):
        raise TypeError("traces must be a list.")

    groups = defaultdict(list)

    for trace in traces:
        strategy = trace.get("strategy", "unknown")
        split = trace.get("split", "unknown")

        groups[(strategy, split)].append(trace)

    summary = {}

    for (strategy, split), items in sorted(groups.items()):
        total = len(items)

        hit1_count = sum(
            bool(item.get("hit_at_1", False))
            for item in items
        )

        hit3_count = sum(
            bool(item.get("evidence_hit_at_3", False))
            for item in items
        )

        summary.setdefault(strategy, {})
        summary[strategy][split] = {
            "total_questions": total,
            "hit_at_1": {
                "numerator": hit1_count,
                "denominator": total,
                "rate": round(hit1_count / total, 4) if total else 0.0
            },
            "evidence_hit_at_3": {
                "numerator": hit3_count,
                "denominator": total,
                "rate": round(hit3_count / total, 4) if total else 0.0
            }
        }

    return summary
