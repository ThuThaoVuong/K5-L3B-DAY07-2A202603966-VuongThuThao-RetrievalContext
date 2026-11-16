
"""Read and normalize source data.

Extract physical PDF pages and normalize text while preserving
useful paragraph and line-break structure.

OCR is intentionally handled as a separate pipeline step.
"""

from pathlib import Path
import logging
import re
import unicodedata

import fitz  # PyMuPDF

from .models import DocumentPage


logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding structure.

    - Normalize Unicode using NFC.
    - Normalize line endings.
    - Remove null characters and form-feed characters.
    - Normalize horizontal whitespace.
    - Preserve paragraph boundaries and line breaks.
    - Remove excessive blank lines.
    """

    if not text:
        return ""

    # Normalize Unicode without compatibility transformations.
    text = unicodedata.normalize("NFC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove characters that are not useful in extracted text.
    text = text.replace("\x00", "")
    text = text.replace("\f", "\n")

    # Normalize non-breaking spaces.
    text = text.replace("\u00a0", " ")

    normalized_lines = []

    for line in text.split("\n"):
        # Normalize horizontal whitespace without removing line breaks.
        line = re.sub(r"[^\S\n]+", " ", line)
        line = line.strip()

        normalized_lines.append(line)

    text = "\n".join(normalized_lines)

    # Preserve paragraph separation but remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(
    path: Path,
    document_id: str,
    source: str
) -> list[DocumentPage]:
    """Extract every physical PDF page.

    Each page becomes one DocumentPage.

    Pages without extractable text are retained with text="",
    allowing the pipeline to identify pages that may need OCR.

    Page numbering is 1-indexed physical PDF page numbering.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a PDF file, got: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Expected a PDF file."
        )

    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    if not source or not source.strip():
        raise ValueError("source cannot be empty.")

    pages: list[DocumentPage] = []
    empty_pages: list[int] = []

    try:
        with fitz.open(path) as pdf:

            if pdf.is_encrypted:
                raise ValueError(
                    f"PDF is encrypted and cannot be processed: {path}"
                )

            total_pages = len(pdf)

            if total_pages == 0:
                logger.warning(
                    "PDF contains no pages: %s",
                    path
                )
                return []

            for page_index in range(total_pages):

                # Physical PDF page number starts at 1.
                page_number = page_index + 1

                pdf_page = pdf.load_page(page_index)

                # Extract text in reading order.
                extracted_text = pdf_page.get_text(
                    "text",
                    sort=True
                )

                normalized = normalize_text(extracted_text)

                # Record pages without extractable text.
                if not normalized:
                    empty_pages.append(page_number)

                    logger.warning(
                        "Page %d has no extractable text. "
                        "OCR may be required. Document: %s",
                        page_number,
                        document_id
                    )

                # Keep even empty pages for traceability.
                document_page = DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=page_number,
                    text=normalized,
                )

                pages.append(document_page)

    except (FileNotFoundError, ValueError):
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Failed to ingest PDF '{path}': {exc}"
        ) from exc

    logger.info(
        "PDF ingestion completed: document_id=%s, "
        "total_pages=%d, empty_pages=%s",
        document_id,
        len(pages),
        empty_pages
    )

    return pages
