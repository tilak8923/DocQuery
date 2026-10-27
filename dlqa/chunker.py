"""Chunker: turns a Document's blocks into retrieval-sized passages.

Rules
- Headings travel with their text: the first heading of a chunk is its
  prefix line and later headings stay inline, so the retriever and reader
  both see the context ("Education" + the degree line beats the degree line
  alone). Chunks are still filled to the target size, because very short
  passages make the reader overconfident.
- Table rows and list items are never cut in half.
- A single block longer than the limit is split by a sliding word window
  with overlap, so an answer on the boundary still appears whole once.
- Each chunk remembers its document, page and section for citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .loader import Block


@dataclass
class Chunk:
    id: int
    text: str
    doc: str = ""
    page: int | None = None
    section: str = ""
    start_word: int = 0
    headings: list[tuple[int, str]] = field(default_factory=list)  # (char offset in text, heading)

    def section_at(self, char_pos: int) -> str:
        """Heading in effect at a character position inside this chunk's text."""
        current = self.section
        for off, h in self.headings:
            if off <= char_pos:
                current = h
        return current


def chunk_blocks(
    blocks: list[Block],
    doc_name: str = "",
    chunk_words: int = 180,
    overlap_words: int = 40,
    start_id: int = 0,
) -> list[Chunk]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    last_heading = ""  # most recent heading seen in the document
    buf: list[Block] = []
    buf_words = 0
    buf_section = ""  # heading in effect when the buffer was started
    word_pos = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if not buf:
            return
        body = "\n".join(b.text for b in buf)
        # Prefix the section unless the buffer already starts with that heading inline.
        prefix = buf_section if buf_section and buf[0].text != buf_section else ""
        text = f"{prefix}\n{body}" if prefix else body
        headings, off = [], len(prefix) + 1 if prefix else 0
        for b in buf:
            if b.kind == "heading":
                headings.append((off, b.text))
            off += len(b.text) + 1
        chunks.append(
            Chunk(
                id=start_id + len(chunks),
                text=text,
                doc=doc_name,
                page=buf[0].page,
                section=buf_section,
                start_word=word_pos - buf_words,
                headings=headings,
            )
        )
        buf, buf_words = [], 0

    for b in blocks:
        n = len(b.text.split())
        if b.kind == "heading":
            # Headings stay inline (the retriever and reader see them) but do not
            # force a new chunk: tiny per-section chunks make the reader overconfident.
            if buf_words + n > chunk_words:
                flush()
            last_heading = b.text
            if buf:  # mid-chunk heading: keep it inline
                buf.append(b)
                buf_words += n
            # else: it becomes the prefix of the next chunk via buf_section
            word_pos += n
            continue
        if n > chunk_words:
            flush()
            words = b.text.split()
            step = chunk_words - overlap_words
            start = 0
            while start < len(words):
                piece = " ".join(words[start : start + chunk_words])
                text = f"{last_heading}\n{piece}" if last_heading else piece
                chunks.append(
                    Chunk(
                        id=start_id + len(chunks),
                        text=text,
                        doc=doc_name,
                        page=b.page,
                        section=last_heading,
                        start_word=word_pos + start,
                    )
                )
                if start + chunk_words >= len(words):
                    break
                start += step
            word_pos += n
            continue
        if buf_words + n > chunk_words:
            flush()
        if not buf:
            buf_section = last_heading
        buf.append(b)
        buf_words += n
        word_pos += n
    flush()
    return chunks


def chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    """Plain sliding-window chunking of raw text (kept for experiments/tests)."""
    if overlap_words >= chunk_words:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()
    chunks: list[Chunk] = []
    step = chunk_words - overlap_words
    start = 0
    while start < len(words):
        piece = words[start : start + chunk_words]
        chunks.append(Chunk(id=len(chunks), text=" ".join(piece), start_word=start))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks
