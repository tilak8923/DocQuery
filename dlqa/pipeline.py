"""Pipeline: load -> chunk -> retrieve -> read (-> generate).

    qa = DocumentQA()
    qa.add_document(load_document("manual.pdf"))
    qa.add_document(load_document("syllabus.docx"))
    r = qa.ask("How long is the warranty?")
    print(r.answer, r.doc, r.page, r.section)

Generative mode (lazy-loads Flan-T5 on first use):
    r = qa.ask("How long is the warranty?", generative=True)
    print(r.generated)   # a full sentence grounded in the retrieved passages
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_blocks
from .config import settings
from .loader import Document, load_bytes, load_document
from .reader import ExtractiveReader
from .retriever import DenseRetriever, RetrievedChunk, TfidfRetriever


@dataclass
class QAResult:
    question: str
    answer: str  # extractive span, "" when nothing found
    confidence: float
    evidence: str  # the chunk the answer came from
    evidence_chunk_id: int
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    seconds: float = 0.0
    doc: str = ""
    page: int | None = None
    section: str = ""
    generated: str | None = None  # generative-mode sentence, None when mode is off
    retrieval_query: str = ""  # what was actually embedded (differs for follow-ups)

    @property
    def found(self) -> bool:
        return bool(self.answer) or bool(self.generated)

    @property
    def citation(self) -> str:
        parts = [self.doc] if self.doc else []
        if self.page:
            parts.append(f"page {self.page}")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)


_PRONOUN_RE = re.compile(r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|there)\b", re.I)


def is_follow_up(question: str) -> bool:
    """Short question that leans on a pronoun -> probably refers to the previous turn."""
    words = question.split()
    return len(words) <= 6 and bool(_PRONOUN_RE.search(question))


class DocumentQA:
    def __init__(
        self,
        retriever: str = "dense",
        chunk_words: int | None = None,
        overlap_words: int | None = None,
        top_k: int | None = None,
        min_confidence: float = 0.6,  # below this the reader is guessing (eval: wrong answers < 0.6, right ones > 0.7)
        null_margin: float = 0.0,
        reader: ExtractiveReader | None = None,
    ):
        self.chunk_words = chunk_words or settings.chunk_words
        self.overlap_words = overlap_words if overlap_words is not None else settings.overlap_words
        self.top_k = top_k or settings.top_k
        self.min_confidence = min_confidence
        self.retriever = DenseRetriever() if retriever == "dense" else TfidfRetriever()
        self.reader = reader or ExtractiveReader(null_margin=null_margin)
        self._generator = None
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []

    # ---- generator (lazy) --------------------------------------------
    @property
    def generator(self):
        if self._generator is None:
            from .generator import Generator

            self._generator = Generator()
        return self._generator

    # ---- documents ----------------------------------------------------
    @property
    def text(self) -> str:
        return "\n\n".join(d.text for d in self.documents)

    def clear(self) -> None:
        self.documents, self.chunks = [], []

    def load(self, path: str) -> int:
        """Replace everything with one document (CLI / evaluation convenience)."""
        self.clear()
        return self.add_document(load_document(path))

    def load_bytes(self, data: bytes, filename: str) -> int:
        self.clear()
        return self.add_document(load_bytes(data, filename))

    def add_document(self, doc: Document) -> int:
        if doc.needs_ocr:
            raise ValueError(
                f"{doc.name} has no extractable text (scanned?) and OCR is not available. "
                "Install Tesseract to read it."
            )
        if not doc.blocks:
            raise ValueError(f"{doc.name} contains no readable text")
        cap = settings.max_index_words
        if cap and doc.words > cap:
            kept, total = [], 0
            for b in doc.blocks:
                n = len(b.text.split())
                if total + n > cap:
                    break
                kept.append(b)
                total += n
            doc.notes.append(
                f"{doc.name} has {doc.words:,} words; only the first {total:,} were indexed on this "
                "host (DLQA_MAX_INDEX_WORDS). Run locally or in Docker for the whole document."
            )
            doc.blocks = kept
        self.documents = [d for d in self.documents if d.name != doc.name] + [doc]
        self._reindex()
        return len([c for c in self.chunks if c.doc == doc.name])

    def remove_document(self, name: str) -> None:
        self.documents = [d for d in self.documents if d.name != name]
        self._reindex()

    def _reindex(self) -> None:
        chunks: list[Chunk] = []
        for d in self.documents:
            chunks += chunk_blocks(d.blocks, d.name, self.chunk_words, self.overlap_words, start_id=len(chunks))
        self.chunks = chunks
        if chunks:
            self.retriever.index_chunks(chunks)

    # ---- asking -------------------------------------------------------
    def ask(
        self,
        question: str,
        generative: bool = False,
        previous_question: str | None = None,
    ) -> QAResult:
        if not self.chunks:
            raise RuntimeError("add a document first")
        t0 = time.perf_counter()

        # Follow-up handling: "what does it cost?" after "tell me about the warranty"
        # retrieves with both questions so the pronoun has something to point at.
        query = question
        if previous_question and is_follow_up(question):
            query = f"{previous_question} {question}"
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)

        best = None
        for rc in retrieved:
            ans = self.reader.answer(question, rc.chunk.text)
            if not ans.text:
                continue
            if best is None or ans.score > best[0].score:
                best = (ans, rc)

        if best is None or best[0].score < self.min_confidence:
            top = retrieved[0].chunk if retrieved else None
            result = QAResult(
                question=question,
                answer="",
                confidence=best[0].score if best else 0.0,
                evidence=top.text if top else "",
                evidence_chunk_id=top.id if top else -1,
                retrieved=retrieved,
                doc=top.doc if top else "",
                page=top.page if top else None,
                section=top.section if top else "",
                retrieval_query=query,
            )
        else:
            ans, rc = best
            result = QAResult(
                question=question,
                answer=ans.text,
                confidence=ans.score,
                evidence=rc.chunk.text,
                evidence_chunk_id=rc.chunk.id,
                retrieved=retrieved,
                doc=rc.chunk.doc,
                page=rc.chunk.page,
                section=rc.chunk.section_at(ans.start_char),
                retrieval_query=query,
            )

        if generative:
            # Grounded generation: only rewrite when the extractive reader found a
            # span. Letting the generator answer on its own produced confident
            # fabrications for unanswerable questions.
            if result.answer:
                contexts = [result.evidence] + [
                    rc.chunk.text for rc in retrieved[:2] if rc.chunk.text != result.evidence
                ]
                result.generated = self.generator.answer(question, contexts, span_hint=result.answer)
            else:
                result.generated = ""

        result.seconds = time.perf_counter() - t0
        return result

    # ---- overview -----------------------------------------------------
    def summarize(self, doc_name: str | None = None) -> str:
        """Summary of one document, or of each loaded document in turn."""
        docs = [d for d in self.documents if doc_name in (None, d.name)]
        parts = []
        for d in docs:
            summary = self.generator.summarize(d.text)
            headings = list(dict.fromkeys(b.text for b in d.blocks if b.kind == "heading"))
            if headings:
                summary += f"\n\n**Sections:** {', '.join(headings[:12])}{' …' if len(headings) > 12 else ''}"
            parts.append(f"**{d.name}** — {summary}" if len(docs) > 1 else summary)
        return "\n\n".join(parts)

    def suggest_questions(self, doc_name: str | None = None, n: int = 4, min_confidence: float = 0.6) -> list[str]:
        """Generate candidate questions and keep only those the reader can actually
        answer from the document, so every suggestion chip works when clicked."""
        text = self._doc_text(doc_name)
        if not text:
            return []
        keep = []
        for q in self.generator.suggest_questions(text, n=n * 3):
            if "of the following" in q.lower():  # quiz-style questions make no sense here
                continue
            r = self.ask(q)
            if r.answer and r.confidence >= min_confidence:
                keep.append(q)
            if len(keep) == n:
                break
        return keep

    def _doc_text(self, doc_name: str | None) -> str:
        if doc_name:
            for d in self.documents:
                if d.name == doc_name:
                    return d.text
            return ""
        return self.text
