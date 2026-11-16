
"""Improvement 2: TF-IDF + Cross-Encoder Reranking."""

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from baseline import hit, question_hit, pack_context, validate
from retrieval_methods import TfidfRetriever


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)


def rerank(question, candidates, model, top_k):
    if not candidates:
        return []

    pairs = [
        (question, chunk["text"])
        for chunk in candidates
    ]

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    ranked = []

    for chunk, score in zip(candidates, scores):
        ranked.append({
            **chunk,
            "retrieval_score": float(chunk["score"]),
            "rerank_score": float(score),
            "method": "tfidf_cross_encoder"
        })

    ranked.sort(
        key=lambda x: (-x["rerank_score"], x["id"])
    )

    return [
        {**chunk, "rank": i + 1}
        for i, chunk in enumerate(ranked[:top_k])
    ]


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=2400
    )

    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "prepared"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "retrieval" / "improvement2"
    )

    args = parser.parse_args()

    if args.candidate_k < args.top_k:
        raise SystemExit("candidate-k must >= top-k")

    if args.top_k < 3 or args.budget <= 0:
        raise SystemExit("Invalid top-k or budget")

    # Load data
    chunks = json.loads(
        (args.data_dir / "chunks.json").read_text(
            encoding="utf-8"
        )
    )

    all_questions = json.loads(
        (args.data_dir / "questions_answerable.json").read_text(
            encoding="utf-8"
        )
    )

    questions = [
        q for q in all_questions
        if q["split"] == args.split
    ]

    if not questions:
        raise SystemExit("No questions found")

    validate(chunks, questions)

    # Initialize models
    print("Loading TF-IDF retriever...")
    lexical = TfidfRetriever(chunks)

    print("Loading Cross-Encoder...")
    reranker = CrossEncoder(
        args.rerank_model,
        max_length=512
    )

    traces = []

    for q in questions:

        started = time.perf_counter()

        # Stage 1: Retrieve top-20 candidates
        candidates = lexical.search(
            q["question"],
            filters=q.get("filter"),
            top_k=args.candidate_k
        )

        retrieval_done = time.perf_counter()

        # Stage 2: Rerank candidates
        results = rerank(
            q["question"],
            candidates,
            reranker,
            args.top_k
        )

        rerank_done = time.perf_counter()

        # Evaluate candidates (offline only)
        candidate_recall = question_hit(
            candidates, q
        )

        marked = [
            {
                **r,
                "hit": hit(r, q["gold"])
            }
            for r in results
        ]

        # Top-3 evidence
        evidence_hit3 = question_hit(
            marked[:3], q
        )

        # Build final context
        context, context_ids = pack_context(
            marked,
            budget=args.budget
        )

        context_rows = [
            r for r in marked
            if r["id"] in context_ids
        ]

        context_evidence = question_hit(
            context_rows, q
        )

        elapsed = (
            rerank_done - started
        ) * 1000

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "method": "tfidf_cross_encoder",
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "gold": q["gold"],
            "candidate_recall": candidate_recall,
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "hit1": question_hit(marked[:1], q),
            "evidence_hit3": evidence_hit3,
            "context_evidence": context_evidence,
            "context_chars": len(context),
            "retrieval_ms": (
                retrieval_done - started
            ) * 1000,
            "rerank_ms": (
                rerank_done - retrieval_done
            ) * 1000,
            "elapsed_ms": elapsed
        })

    # Aggregate metrics
    n = len(traces)

    def metric(key):
        count = sum(t[key] for t in traces)
        return {
            "n": count,
            "N": n,
            "rate": count / n
        }

    summary = {
        "experiment": "improvement_2",
        "description": (
            "TF-IDF top-20 + Cross-Encoder reranking, "
            "final top-5, context budget=2400"
        ),
        "method": "tfidf_cross_encoder",
        "split": args.split,
        "n": n,
        "candidate_recall": metric("candidate_recall"),
        "hit1": metric("hit1"),
        "evidence_hit3": metric("evidence_hit3"),
        "context_evidence_rate": metric(
            "context_evidence"
        ),
        "median_ms": statistics.median(
            t["elapsed_ms"] for t in traces
        ),
        "median_retrieval_ms": statistics.median(
            t["retrieval_ms"] for t in traces
        ),
        "median_rerank_ms": statistics.median(
            t["rerank_ms"] for t in traces
        ),
        "avg_context_chars": statistics.mean(
            t["context_chars"] for t in traces
        ),
        "rerank_model": args.rerank_model,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "context_budget_chars": args.budget,
        "service_cost_usd": 0
    }

    # Save results
    output = (
        args.output_dir
        / args.split
        / "tfidf_cross_encoder"
    )

    output.mkdir(parents=True, exist_ok=True)

    (output / "traces.json").write_text(
        json.dumps(
            traces,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    (output / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()
