"""
Chunking strategy: semantic/structural first (split on section headers,
since policy documents are naturally clause-based), then fall back to
fixed-size + overlap for any section that's still too long.

This mirrors the "recursive chunking" idea from the RAG pipeline:
try the most meaningful split first, fall back to a cruder one only
when necessary.
"""
import re
from dataclasses import dataclass

SECTION_HEADER_RE = re.compile(r"^(Section\s+[\d.]+.*)$", re.MULTILINE)

MAX_CHUNK_CHARS = 1200      # roughly 250-300 tokens
OVERLAP_CHARS = 150         # ~10-15% overlap for fallback splitting


@dataclass
class Chunk:
    section: str
    content: str


def split_into_sections(raw_text: str) -> list[Chunk]:
    """
    Split a document on 'Section X.Y ...' headers. Each section header
    plus the text that follows (until the next header) becomes one
    semantic chunk.
    """
    matches = list(SECTION_HEADER_RE.finditer(raw_text))
    if not matches:
        # No section structure found -- treat the whole doc as one "section"
        return _fallback_split("Unstructured", raw_text)

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if not body:
            continue

        full_section_text = f"{header}\n{body}"
        if len(full_section_text) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(section=header, content=full_section_text))
        else:
            # Section itself is too long for one chunk -- fall back to
            # size-based splitting, but keep the header on every piece
            # so retrieval never loses the "which section is this" context.
            chunks.extend(_fallback_split(header, full_section_text))

    return chunks


def _fallback_split(section_label: str, text: str) -> list[Chunk]:
    """Fixed-size splitting with overlap, used only when structural
    splitting isn't available or a section is too long on its own."""
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(section=section_label, content=piece))
        if end == len(text):
            break
        start = end - OVERLAP_CHARS  # step back to create overlap
    return chunks
