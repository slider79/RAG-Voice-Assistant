"""Document Loader Tool: read PDF / TXT / DOCX and split into chunks.

Extraction is kept dependency-light: pypdf for PDFs, python-docx for Word, and
plain decoding for text. Chunking uses a simple sliding window over words with
overlap, which is enough for retrieval and easy to reason about.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

SUPPORTED = (".pdf", ".txt", ".md", ".docx")

# Chunk sizing in words. ~180 words is roughly a paragraph, small enough to be
# a precise retrieval unit but large enough to carry context. Overlap keeps a
# sentence that straddles a boundary retrievable from either chunk.
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40


class DocumentError(RuntimeError):
    pass


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from a file's bytes based on its extension."""
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            return _extract_pdf(data)
        if name.endswith(".docx"):
            return _extract_docx(data)
        if name.endswith((".txt", ".md")):
            return data.decode("utf-8", errors="replace")
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentError(f"Could not read {filename}: {exc}") from None
    raise DocumentError(f"Unsupported file type: {filename}. Use PDF, TXT, MD or DOCX.")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split text into overlapping word windows, dropping empty ones."""
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    idx = 0
    for start in range(0, len(words), step):
        window = words[start : start + CHUNK_WORDS]
        piece = " ".join(window).strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, index=idx))
            idx += 1
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def load_and_chunk(filename: str, data: bytes) -> list[Chunk]:
    """Full pipeline for one file: extract, then chunk."""
    text = extract_text(filename, data)
    chunks = chunk_text(text, filename)
    if not chunks:
        raise DocumentError(f"No readable text found in {filename}.")
    return chunks
