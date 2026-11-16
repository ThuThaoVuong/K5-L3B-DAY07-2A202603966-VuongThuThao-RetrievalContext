
"""Store and search vectors using PostgreSQL + pgvector.

Supports:
- Separate vector collections for strategies A, B and C.
- Stable chunk IDs and document-level replacement.
- Cosine similarity search.
- Parameterized metadata filtering.
- Source/page traceability.
"""

from typing import Any
from collections import defaultdict
import json
import math

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from .models import Chunk


class PgVectorStore:
    """PostgreSQL vector store for RAG experiments."""

    TABLE_NAME = "rag_chunks"

    VALID_STRATEGIES = {"A", "B", "C"}

    def __init__(
        self,
        database_url: str,
        dimensions: int
    ) -> None:
        """Retain connection settings and validate embedding dimensions."""

        if not database_url or not database_url.strip():
            raise ValueError("database_url cannot be empty.")

        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer.")

        self.database_url = database_url
        self.dimensions = dimensions

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    def _connect(self):
        """Create a PostgreSQL connection and register pgvector."""

        conn = psycopg.connect(self.database_url)
        register_vector(conn)

        return conn

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _validate_strategy(self, strategy: str) -> None:
        """Ensure that the requested strategy is supported."""

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                "Choose A, B or C."
            )

    def _validate_vector(self, vector: list[float]) -> None:
        """Validate embedding dimensions and numeric values."""

        if not isinstance(vector, (list, tuple)):
            raise TypeError("Embedding must be a list or tuple.")

        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected embedding dimension {self.dimensions}, "
                f"got {len(vector)}."
            )

        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Embedding values must be numeric.")

            if not math.isfinite(value):
                raise ValueError(
                    "Embedding contains NaN or infinite values."
                )

    # --------------------------------------------------------
    # CREATE SCHEMA
    # --------------------------------------------------------

    def create_schema(self) -> None:
        """Create pgvector extension, table and vector index."""

        create_extension_sql = """
            CREATE EXTENSION IF NOT EXISTS vector;
        """

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                strategy TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding VECTOR({self.dimensions}) NOT NULL,

                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                PRIMARY KEY (strategy, chunk_id)
            );
        """

        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS
                idx_{self.TABLE_NAME}_embedding
            ON {self.TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops);
        """

        create_metadata_index_sql = f"""
            CREATE INDEX IF NOT EXISTS
                idx_{self.TABLE_NAME}_metadata
            ON {self.TABLE_NAME}
            USING gin (metadata);
        """

        create_lookup_index_sql = f"""
            CREATE INDEX IF NOT EXISTS
                idx_{self.TABLE_NAME}_document
            ON {self.TABLE_NAME}
                (strategy, document_id);
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_extension_sql)
                cursor.execute(create_table_sql)
                cursor.execute(create_index_sql)
                cursor.execute(create_metadata_index_sql)
                cursor.execute(create_lookup_index_sql)

        # _connect() registers the vector adapter
        # for subsequent database operations.

    # --------------------------------------------------------
    # UPSERT
    # --------------------------------------------------------

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]]
    ) -> None:
        """Insert or replace chunks and their embeddings.

        Existing chunks belonging to incoming documents are removed
        before inserting the new version.

        This ensures obsolete chunks do not remain after re-ingestion.

        IMPORTANT:
        Pass all chunks belonging to each document being replaced.
        """

        self._validate_strategy(strategy)

        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list.")

        if not isinstance(vectors, list):
            raise TypeError("vectors must be a list.")

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) does not match "
                f"number of vectors ({len(vectors)})."
            )

        if not chunks:
            return

        seen_ids = set()

        for chunk, vector in zip(chunks, vectors):

            if not isinstance(chunk, Chunk):
                raise TypeError(
                    "Each item in chunks must be a Chunk instance."
                )

            if chunk.chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk_id: {chunk.chunk_id}"
                )

            seen_ids.add(chunk.chunk_id)

            self._validate_vector(vector)

            if not chunk.text.strip():
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has empty text."
                )

        # Group chunks by their original documents.
        document_chunks = defaultdict(list)

        for chunk, vector in zip(chunks, vectors):
            document_chunks[chunk.document_id].append(
                (chunk, vector)
            )

        insert_sql = f"""
            INSERT INTO {self.TABLE_NAME} (
                strategy,
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            ON CONFLICT (strategy, chunk_id)
            DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                created_at = NOW();
        """

        delete_sql = f"""
            DELETE FROM {self.TABLE_NAME}
            WHERE strategy = %s
              AND document_id = %s;
        """

        with self._connect() as conn:
            with conn.cursor() as cursor:

                for document_id, items in document_chunks.items():

                    # Remove obsolete chunks of this document
                    # for this strategy only.
                    cursor.execute(
                        delete_sql,
                        (strategy, document_id)
                    )

                    rows = []

                    for chunk, vector in items:

                        metadata = dict(chunk.metadata or {})

                        rows.append((
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.text,
                            json.dumps(
                                metadata,
                                ensure_ascii=False
                            ),
                            Vector(vector)
                        ))

                    cursor.executemany(
                        insert_sql,
                        rows
                    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search nearest chunks using cosine distance.

        Supported filters:
            source
            document_id
            page
            Any additional metadata field, e.g. section.

        Metadata filters are applied in SQL before ranking.

        Returns:
            [
                {
                    "chunk_id": "...",
                    "text": "...",
                    "score": 0.85,
                    "metadata": {
                        "source": "...",
                        "page": 5,
                        "document_id": "...",
                        "section": "..."
                    }
                }
            ]
        """

        self._validate_strategy(strategy)
        self._validate_vector(query_vector)

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None.")

        filters = filters or {}

        conditions = ["strategy = %s"]
        params: list[Any] = [strategy]

        # Filter by dedicated SQL columns.
        column_filters = {
            "source": "source",
            "document_id": "document_id",
            "page": "page"
        }

        metadata_filters = {}

        for key, value in filters.items():

            if value is None:
                continue

            if key in column_filters:
                column = column_filters[key]

                conditions.append(f"{column} = %s")
                params.append(value)

            else:
                # Additional fields are stored in JSONB metadata.
                metadata_filters[key] = value

        # Apply metadata filtering using JSONB containment.
        if metadata_filters:

            conditions.append(
                "metadata @> %s::jsonb"
            )

            params.append(
                json.dumps(
                    metadata_filters,
                    ensure_ascii=False
                )
            )

        where_clause = " AND ".join(conditions)

        search_sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                1 - (embedding <=> %s) AS score

            FROM {self.TABLE_NAME}

            WHERE {where_clause}

            ORDER BY embedding <=> %s

            LIMIT %s;
        """

        # Convert query vectors to pgvector.Vector
        # so PostgreSQL receives the correct vector type.
        final_params = [
            Vector(query_vector),
            *params,
            Vector(query_vector),
            top_k
        ]

        results = []

        with self._connect() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    search_sql,
                    final_params
                )

                rows = cursor.fetchall()

                for row in rows:

                    (
                        chunk_id,
                        document_id,
                        source,
                        page,
                        text,
                        metadata,
                        score
                    ) = row

                    # Combine original source metadata
                    # with additional chunk metadata.
                    result_metadata = dict(metadata or {})

                    result_metadata.update({
                        "document_id": document_id,
                        "source": source,
                        "page": page
                    })

                    results.append({
                        "chunk_id": chunk_id,
                        "text": text,
                        "score": float(score),
                        "metadata": result_metadata
                    })

        return results
