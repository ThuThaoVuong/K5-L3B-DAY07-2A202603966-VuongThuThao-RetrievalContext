
"""Run comparable retrieval experiments over Day 6's same corpus/questions."""
from __future__ import annotations

import argparse
import json
import os
import time
import statistics
from pathlib import Path

from dotenv import load_dotenv

from day6_adapter import Day6DenseRetriever
from retrieval_methods import TfidfRetriever, RetrievalRouter
from baseline import hit, question_hit, pack_context, validate


ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True
    )

    parser.add_argument(
        "--method",
        choices=["dense", "tfidf", "hybrid", "all"],
        default="tfidf"
    )

    parser.add_argument(
        "--strategy",
        choices=["A", "B", "C"],
        default=os.getenv("DAY6_STRATEGY", "C")
    )

    # Improvement 1: tăng top-k từ 3 lên 5
    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    # Improvement 1: tăng context budget từ 1800 lên 2400
    parser.add_argument(
        "--budget",
        type=int,
        default=2400
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "prepared"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "retrieval" / "improvement1"
    )

    args = parser.parse_args()

    if args.top_k < 3:
        raise SystemExit("--top-k must be at least 3 for comparable evaluation")

    if args.budget <= 0:
        raise SystemExit("--budget must be greater than 0")

    # Load database configuration
    db = os.getenv("DATABASE_URL")

    if not db:
        raise SystemExit("DATABASE_URL is missing in .env")

    # Load chunks and answerable questions
    chunks_path = args.data_dir / "chunks.json"
    questions_path = args.data_dir / "questions_answerable.json"

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    all_questions = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    questions = [
        q for q in all_questions
        if q["split"] == args.split
    ]

    if not questions:
        raise SystemExit(
            f"No questions for split={args.split}"
        )

    validate(chunks, questions)

    # Initialize retrievers
    dense = Day6DenseRetriever(
        db,
        args.strategy,
        os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    lexical = TfidfRetriever(chunks)
    router = RetrievalRouter(dense, lexical)

    methods = (
        ["tfidf"]
        if args.method == "tfidf"
        else ["dense", "tfidf", "hybrid"]
        if args.method == "all"
        else [args.method]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for method in methods:

        traces = []

        for q in questions:

            started = time.perf_counter()

            # Retrieve top-5 chunks
            results = router.search(
                method,
                q["question"],
                filters=q.get("filter"),
                top_k=args.top_k
            )

            elapsed = (
                time.perf_counter() - started
            ) * 1000

            marked = [
                {
                    **r,
                    "hit": hit(r, q["gold"])
                }
                for r in results
            ]

            # Keep top-3 for comparable ranking evaluation
            top3 = marked[:3]

            # Build context using up to top-5 retrieved chunks
            context, context_ids = pack_context(
                marked,
                budget=args.budget
            )

            context_rows = [
                r for r in marked
                if r["id"] in context_ids
            ]

            traces.append({
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "method": method,
                "strategy": args.strategy,
                "embedding_model": dense.model_name,
                "embedding_dimensions": dense.dimensions,
                "top_k": args.top_k,
                "filter": q.get("filter"),
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": context_ids,

                "hit1": question_hit(
                    marked[:1], q
                ),

                "evidence_hit3": question_hit(
                    top3, q
                ),

                "context_evidence": question_hit(
                    context_rows, q
                ),

                "context_chars": len(context),
                "elapsed_ms": elapsed
            })

        n = len(traces)

        hit1_count = sum(
            t["hit1"] for t in traces
        )

        evidence_hit3_count = sum(
            t["evidence_hit3"] for t in traces
        )

        context_evidence_count = sum(
            t["context_evidence"] for t in traces
        )

        summary = {
            "experiment": "improvement_1",
            "description": (
                "TF-IDF retrieval with top-k=5 "
                "and context budget=2400"
            ),
            "method": method,
            "split": args.split,
            "n": n,

            "hit1": {
                "n": hit1_count,
                "N": n,
                "rate": hit1_count / n
            },

            "evidence_hit3": {
                "n": evidence_hit3_count,
                "N": n,
                "rate": evidence_hit3_count / n
            },

            "context_evidence_rate": {
                "n": context_evidence_count,
                "N": n,
                "rate": context_evidence_count / n
            },

            "median_ms": statistics.median(
                t["elapsed_ms"] for t in traces
            ),

            "avg_context_chars": statistics.mean(
                t["context_chars"] for t in traces
            ),

            "embedding_model": dense.model_name,
            "embedding_dimensions": dense.dimensions,
            "vector_strategy": args.strategy,
            "top_k": args.top_k,
            "context_budget_chars": args.budget,
            "service_cost_usd": 0
        }

        output_path = (
            args.output_dir
            / args.split
            / method
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        (output_path / "traces.json").write_text(
            json.dumps(
                traces,
                ensure_ascii=False,
                indent=2
            ) + "\n",
            encoding="utf-8"
        )

        (output_path / "summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
            ) + "\n",
            encoding="utf-8"
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
            )
        )


if __name__ == "__main__":
    main()
