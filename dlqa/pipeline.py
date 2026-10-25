"""Pipeline: load -> chunk -> retrieve -> read.

Usage:
    qa = DocumentQA()
    qa.load("paper.pdf")
    result = qa.ask("What dataset was used?")
    print(result.answer, result.evidence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_text
from .loader import load_bytes, load_document
from .reader import ExtractiveReader
from .retriever import DenseRetriever, RetrievedChunk, TfidfRetriever


@dataclass
class QAResult:
    question: str
    answer: str  # "" when nothing found
    confidence: float
    evidence: str  # the chunk the answer came from
    evidence_chunk_id: int
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.answer)


class DocumentQA:
    def __init__(
        self,
        retriever: str = "dense",
        chunk_words: int = 180,
        overlap_words: int = 40,
        top_k: int = 5,  # chosen by scripts/sweep.py: Recall@5 = 1.00 on the eval sets
        min_confidence: float = 0.05,
        null_margin: float = 0.0,  # >0 answers more often but breaks abstaining; see data/eval/sweep.json
        reader: ExtractiveReader | None = None,
    ):
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.retriever = DenseRetriever() if retriever == "dense" else TfidfRetriever()
        # Allow sharing one loaded reader across pipelines (used by sweeps).
        self.reader = reader or ExtractiveReader(null_margin=null_margin)
        self.chunks: list[Chunk] = []
        self.text: str = ""

    # ---- indexing -----------------------------------------------------
    def load(self, path: str) -> int:
        return self._index(load_document(path))

    def load_bytes(self, data: bytes, filename: str) -> int:
        return self._index(load_bytes(data, filename))

    def _index(self, text: str) -> int:
        if not text.strip():
            raise ValueError("Document contains no extractable text")
        self.text = text
        self.chunks = chunk_text(text, self.chunk_words, self.overlap_words)
        self.retriever.index_chunks(self.chunks)
        return len(self.chunks)

    # ---- asking -------------------------------------------------------
    def ask(self, question: str) -> QAResult:
        if not self.chunks:
            raise RuntimeError("load() a document first")
        t0 = time.perf_counter()

        retrieved = self.retriever.retrieve(question, top_k=self.top_k)

        # Run the reader on each retrieved chunk and keep the best non-empty answer.
        best = None
        for rc in retrieved:
            ans = self.reader.answer(question, rc.chunk.text)
            if not ans.text:
                continue
            if best is None or ans.score > best[0].score:
                best = (ans, rc)

        seconds = time.perf_counter() - t0
        if best is None or best[0].score < self.min_confidence:
            top = retrieved[0].chunk if retrieved else None
            return QAResult(
                question=question,
                answer="",
                confidence=best[0].score if best else 0.0,
                evidence=top.text if top else "",
                evidence_chunk_id=top.id if top else -1,
                retrieved=retrieved,
                seconds=seconds,
            )

        ans, rc = best
        return QAResult(
            question=question,
            answer=ans.text,
            confidence=ans.score,
            evidence=rc.chunk.text,
            evidence_chunk_id=rc.chunk.id,
            retrieved=retrieved,
            seconds=seconds,
        )
