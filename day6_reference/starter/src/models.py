
"""Shared data contracts; keep traceability from chunk back to source."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    """Represent one physical page extracted from a source document.

    Attributes:
        document_id: Unique identifier of the original document.
        source: Stable source identifier used by the benchmark.
        page: 1-indexed physical PDF page number.
        text: Normalized text extracted from the page.
        metadata: Additional page-level information, such as section.
    """

    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty.")

        if not self.source.strip():
            raise ValueError("source cannot be empty.")

        if self.page < 1:
            raise ValueError("page must be a positive 1-indexed number.")

        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")


@dataclass
class Chunk:
    """Represent a text chunk with source traceability.

    Each chunk retains its original document ID, source and page.
    Metadata can store additional information such as section,
    chunking strategy and character positions.
    """

    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty.")

        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty.")

        if not self.source.strip():
            raise ValueError("source cannot be empty.")

        if self.page < 1:
            raise ValueError("page must be a positive 1-indexed number.")

        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
