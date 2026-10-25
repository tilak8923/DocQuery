"""Chunker: splits document text into overlapping passages.

Why: the reader transformer accepts about 512 tokens at a time, so the
document has to be cut into pieces small enough to fit. Overlap makes
sure an answer that sits on a boundary still appears whole in one chunk.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    start_word: int  # position in the document, useful for showing evidence


def chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap must be smaller than chunk size")

    words = text.split()
    chunks: list[Chunk] = []
    step = chunk_words - overlap_words
    start = 0
    cid = 0
    while start < len(words):
        piece = words[start : start + chunk_words]
        chunks.append(Chunk(id=cid, text=" ".join(piece), start_word=start))
        cid += 1
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks
