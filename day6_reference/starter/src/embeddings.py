"""Real local embedding provider using Sentence Transformers.

No mock vectors or silent fallback are used. The model name is configurable
so the exact model can be recorded and reused for every strategy.
"""

import math
from typing import Any, Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class MyEmbedder:
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("Embedding model name cannot be empty.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "`python -m pip install -r requirements.txt` from starter/."
            ) from exc

        self.model_name = model_name.strip()
        # Fail loudly if the real model cannot be loaded.
        self.model = SentenceTransformer(self.model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()

        if not isinstance(self.dimensions, int) or self.dimensions <= 0:
            raise RuntimeError("Could not determine a valid embedding dimension.")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every input must be a string.")
        if any(not text.strip() for text in texts):
            raise ValueError("Input texts cannot contain empty strings.")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = embeddings.tolist()

        if len(vectors) != len(texts):
            raise RuntimeError("Embedding output count does not match input count.")

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError("Embedding vector has an unexpected dimension.")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in vector
            ):
                raise RuntimeError("Embedding contains invalid numeric values.")

        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        return self._encode([question])[0]
