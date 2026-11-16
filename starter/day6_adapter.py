"""Adapters that reuse the exact Day 6 embedding model and pgvector store."""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = Path(os.getenv("DAY6_PROJECT_DIR", str(ROOT / "day6_reference"))).resolve()
DAY6_STARTER = DAY6_ROOT / "starter"
if str(DAY6_STARTER) not in sys.path:
    sys.path.insert(0, str(DAY6_STARTER))

from src.embeddings import MyEmbedder
from src.store import PgVectorStore

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class Day6DenseRetriever:
    def __init__(self, database_url: str, strategy: str = "C", model_name: str = DEFAULT_MODEL):
        self.embedder = MyEmbedder(model_name)
        self.store = PgVectorStore(database_url, self.embedder.dimensions)
        self.strategy = strategy
        self.model_name = self.embedder.model_name
        self.dimensions = self.embedder.dimensions

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        vector = self.embedder.embed_query(query)
        rows = self.store.search(self.strategy, vector, top_k=top_k, filters=filters)
        return [normalise_hit(row, rank=i + 1, method="dense") for i, row in enumerate(rows)]


def normalise_hit(row: dict[str, Any], rank: int, method: str):
    meta = dict(row.get("metadata") or {})
    chunk_id = row.get("chunk_id") or row.get("id")
    source = meta.get("source") or row.get("source")
    page = meta.get("page", row.get("page"))
    document_id = meta.get("document_id") or row.get("document_id") or row.get("source_id")
    return {"id": chunk_id, "text": row["text"], "source_id": document_id,
            "source": source, "page": int(page), "metadata": meta,
            "score": float(row.get("score", 0.0)), "rank": rank, "method": method}
