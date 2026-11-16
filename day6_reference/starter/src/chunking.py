
"""Improved chunking strategies A, B and C.

A: fixed-size overlapping windows with word-boundary alignment.
B: structure-aware chunks that preserve headings, paragraphs and lists,
   splitting oversized paragraphs at sentence boundaries.
C: exactly the same text and boundaries as B; metadata filtering is
   intentionally deferred to retrieval time.

All strategies preserve document/page traceability and source metadata.
"""

import hashlib
import re
from typing import Literal

from .models import Chunk, DocumentPage

Strategy = Literal["A", "B", "C"]

DEFAULT_SIZE = 500
DEFAULT_OVERLAP = 50

# Common heading patterns found in handbooks and extracted PDF text.
HEADING_RE = re.compile(
    r"^(?:"
    r"(?:chapter|section|part)\s+[\w.-]+"
    r"|"
    r"\d+(?:\.\d+)*\.?\s+\S.+"
    r"|"
    r"[A-Z][A-Z0-9\s&:/(),'-]{2,}$"
    r")$",
    re.IGNORECASE,
)

BULLET_RE = re.compile(
    r"^\s*(?:[-*•▪◦‣]|\d+[.)]|[a-zA-Z][.)])\s+"
)

SENTENCE_RE = re.compile(
    r".+?(?:[.!?。！？]+(?=\s|$)|$)",
    re.DOTALL,
)


def _validate_size(size: int, overlap: int = 0) -> None:
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("size must be a positive integer.")

    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")

    if overlap >= size:
        raise ValueError("overlap must be smaller than size.")


def _normalize_text(text: str) -> str:
    """Normalize line endings without removing structural line breaks."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _make_chunk(
    page: DocumentPage,
    text: str,
    chunk_index: int,
    strategy: str,
    char_start: int | None = None,
    char_end: int | None = None,
) -> Chunk:
    """Create a deterministic chunk with inherited document metadata."""
    digest_input = (
        f"{page.document_id}|{page.source}|{page.page}|"
        f"{strategy}|{chunk_index}|{text}"
    )
    digest = hashlib.sha1(
        digest_input.encode("utf-8")
    ).hexdigest()[:16]

    chunk_id = (
        f"{page.source}-p{page.page:03d}-"
        f"{strategy}-{chunk_index:04d}-{digest}"
    )

    metadata = dict(page.metadata or {})
    metadata.update({
        "chunk_index": chunk_index,
        "chunking_strategy": strategy,
    })

    if char_start is not None:
        metadata["char_start"] = char_start

    if char_end is not None:
        metadata["char_end"] = char_end

    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


# ---------------------------------------------------------
# Strategy A: Fixed-size with word-boundary alignment
# ---------------------------------------------------------

def _adjust_end(
    text: str,
    start: int,
    target_end: int,
    max_size: int,
) -> int:
    """Move a window boundary to a nearby whitespace boundary."""
    if target_end >= len(text):
        return len(text)

    lower = max(start + 1, target_end - 30)
    upper = min(len(text), start + max_size)

    # Prefer a boundary near the target without exceeding max_size.
    for position in range(target_end, lower - 1, -1):
        if text[position - 1].isspace():
            return position

    for position in range(target_end + 1, upper + 1):
        if position < len(text) and text[position].isspace():
            return position

    return target_end


def fixed_size(
    page: DocumentPage,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a page into overlapping character windows."""
    _validate_size(size, overlap)

    text = _normalize_text(page.text)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    step = size - overlap

    while start < len(text):
        target_end = min(start + size, len(text))

        if target_end == len(text):
            end = target_end
        else:
            end = _adjust_end(text, start, target_end, size)

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                _make_chunk(
                    page,
                    chunk_text,
                    len(chunks),
                    "A",
                    start,
                    end,
                )
            )

        if end >= len(text):
            break

        # Preserve approximate overlap while avoiding an endless loop.
        next_start = max(start + 1, end - overlap)

        # Align the next starting position to a word boundary.
        while (
            next_start < end
            and next_start < len(text)
            and not text[next_start].isspace()
        ):
            next_start += 1

        start = max(next_start, start + step)

    return chunks


# ---------------------------------------------------------
# Strategy B: Structure-aware chunking
# ---------------------------------------------------------

def _is_heading(line: str) -> bool:
    """Identify likely section headings without changing their content."""
    value = line.strip()

    if not value or len(value) > 120:
        return False

    if BULLET_RE.match(value):
        return False

    return bool(HEADING_RE.match(value))


def _line_blocks(text: str) -> list[tuple[int, int]]:
    """Group PDF lines into headings, lists and paragraph-like blocks."""
    lines = text.splitlines(keepends=True)

    blocks: list[tuple[int, int]] = []
    position = 0
    current_start = None
    current_end = None
    current_type = None

    def flush():
        nonlocal current_start, current_end, current_type

        if current_start is not None and current_end is not None:
            blocks.append((current_start, current_end))

        current_start = None
        current_end = None
        current_type = None

    for raw_line in lines:
        line = raw_line.strip()
        line_start = position
        line_end = position + len(raw_line)
        position = line_end

        if not line:
            flush()
            continue

        if _is_heading(line):
            flush()
            blocks.append((line_start, line_end))
            continue

        is_bullet = bool(BULLET_RE.match(line))
        line_type = "bullet" if is_bullet else "paragraph"

        if current_start is None:
            current_start = line_start
            current_end = line_end
            current_type = line_type

        elif line_type == "bullet" and current_type == "bullet":
            # Keep consecutive list items together.
            current_end = line_end

        elif line_type == "paragraph" and current_type == "paragraph":
            # PDF line wrapping: combine continuation lines.
            current_end = line_end

        else:
            flush()
            current_start = line_start
            current_end = line_end
            current_type = line_type

    flush()
    return blocks


def _get_blocks(text: str) -> list[tuple[int, int]]:
    """Extract structural blocks with positions in normalized page text."""
    # First use blank lines, which are the strongest paragraph separator.
    blocks = [
        (match.start(), match.end())
        for match in re.finditer(
            r"\S(?:.*?\S)?(?=\n\s*\n|$)",
            text,
            re.DOTALL,
        )
    ]

    # If the PDF extractor did not preserve paragraph separators,
    # use headings and line/list structure instead.
    if len(blocks) <= 1 and "\n" in text:
        blocks = _line_blocks(text)

    return [
        (start, end)
        for start, end in blocks
        if text[start:end].strip()
    ]


def _sentence_spans(
    text: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Split a long block into sentence spans while retaining offsets."""
    block = text[start:end]
    result = []

    for match in SENTENCE_RE.finditer(block):
        left = match.start()
        right = match.end()

        while left < right and block[left].isspace():
            left += 1

        while right > left and block[right - 1].isspace():
            right -= 1

        if left < right:
            result.append((start + left, start + right))

    if not result and block.strip():
        left = len(block) - len(block.lstrip())
        right = len(block.rstrip())
        result.append((start + left, start + right))

    return result


def _split_oversized_block(
    text: str,
    start: int,
    end: int,
    max_size: int,
) -> list[tuple[int, int]]:
    """Split an oversized paragraph at sentence boundaries.

    If a single sentence is still too long, split at word boundaries.
    """
    sentences = _sentence_spans(text, start, end)

    result: list[tuple[int, int]] = []
    current_start = None
    current_end = None

    for sentence_start, sentence_end in sentences:
        sentence_length = sentence_end - sentence_start

        if sentence_length > max_size:
            if current_start is not None:
                result.append((current_start, current_end))
                current_start = None
                current_end = None

            sentence = text[sentence_start:sentence_end]
            cursor = sentence_start

            while cursor < sentence_end:
                limit = min(cursor + max_size, sentence_end)

                if limit < sentence_end:
                    boundary = text.rfind(" ", cursor, limit)

                    if boundary > cursor:
                        limit = boundary

                result.append((cursor, limit))
                cursor = limit

                while cursor < sentence_end and text[cursor].isspace():
                    cursor += 1

            continue

        if current_start is None:
            current_start = sentence_start
            current_end = sentence_end
            continue

        candidate_length = sentence_end - current_start

        if candidate_length <= max_size:
            current_end = sentence_end
        else:
            result.append((current_start, current_end))
            current_start = sentence_start
            current_end = sentence_end

    if current_start is not None:
        result.append((current_start, current_end))

    return result


def by_paragraph(
    page: DocumentPage,
    max_size: int = DEFAULT_SIZE,
) -> list[Chunk]:
    """Create structure-aware chunks preserving headings and paragraphs."""
    _validate_size(max_size)

    text = _normalize_text(page.text)
    if not text:
        return []

    blocks = _get_blocks(text)

    # Expand oversized blocks into sentence-level spans.
    units: list[tuple[int, int]] = []

    for start, end in blocks:
        block_text = text[start:end].strip()

        if not block_text:
            continue

        if len(block_text) <= max_size:
            units.append((start, end))
        else:
            units.extend(
                _split_oversized_block(
                    text,
                    start,
                    end,
                    max_size,
                )
            )

    chunks: list[Chunk] = []
    current_start = None
    current_end = None

    def flush():
        nonlocal current_start, current_end

        if current_start is None or current_end is None:
            return

        chunk_text = text[current_start:current_end].strip()

        if chunk_text:
            chunks.append(
                _make_chunk(
                    page,
                    chunk_text,
                    len(chunks),
                    "B",
                    current_start,
                    current_end,
                )
            )

        current_start = None
        current_end = None

    for start, end in units:
        unit_text = text[start:end].strip()

        if not unit_text:
            continue

        if current_start is None:
            current_start = start
            current_end = end
            continue

        candidate_length = end - current_start

        if candidate_length <= max_size:
            current_end = end
        else:
            flush()
            current_start = start
            current_end = end

    flush()
    return chunks


# ---------------------------------------------------------
# Strategy C: Same boundaries as B, filtering at retrieval
# ---------------------------------------------------------

def metadata_filtering(
    page: DocumentPage,
    max_size: int = DEFAULT_SIZE,
) -> list[Chunk]:
    """Use B's exact chunk boundaries; do not filter during chunking."""
    b_chunks = by_paragraph(page, max_size=max_size)

    return [
        _make_chunk(
            page,
            chunk.text,
            index,
            "C",
            chunk.metadata.get("char_start"),
            chunk.metadata.get("char_end"),
        )
        for index, chunk in enumerate(b_chunks)
    ]


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def chunk_page(
    page: DocumentPage,
    strategy: Strategy,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a document page using the selected strategy."""
    if strategy == "A":
        return fixed_size(
            page,
            size=size,
            overlap=overlap,
        )

    if strategy == "B":
        return by_paragraph(
            page,
            max_size=size,
        )

    if strategy == "C":
        return metadata_filtering(
            page,
            max_size=size,
        )

    raise ValueError(
        f"Unknown strategy {strategy!r}; choose A, B or C."
    )


def chunk_statistics(chunks: list[Chunk]) -> dict:
    """Return reproducible chunk-size statistics."""
    lengths = [len(chunk.text) for chunk in chunks]

    if not lengths:
        return {
            "chunk_count": 0,
            "total_characters": 0,
            "average_length": 0,
            "min_length": 0,
            "max_length": 0,
        }

    return {
        "chunk_count": len(lengths),
        "total_characters": sum(lengths),
        "average_length": round(
            sum(lengths) / len(lengths),
            2,
        ),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
