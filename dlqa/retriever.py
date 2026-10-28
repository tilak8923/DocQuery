"""Retriever: finds the chunks most relevant to a question.

Two implementations share one interface so they can be compared in the
evaluation:

- DenseRetriever: a sentence-transformer neural network turns every chunk
  and the question into a 384-d vector. Similar meaning -> nearby vectors.
  FAISS does the nearest-neighbour search. This is the deep-learning path.
- TfidfRetriever: classic keyword matching. No neural network. Kept as the
  baseline so we can show the dense model actually helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk
from .config import settings


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class DenseRetriever:
    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embed_model
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.index = None

    def index_chunks(self, chunks: list[Chunk], progress=None) -> None:
        import faiss

        self.chunks = chunks
        texts = [c.text for c in chunks]
        batch = 64
        parts = []
        for i in range(0, len(texts), batch):
            if progress is not None:
                progress(i, len(texts))
            parts.append(
                self.model.encode(
                    texts[i : i + batch],
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # so inner product == cosine similarity
                    show_progress_bar=False,
                )
            )
        embeddings = np.vstack(parts).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("index_chunks() must be called first")
        q = self.model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(q, k)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]


class TfidfRetriever:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: list[Chunk] = []
        self.matrix = None

    def index_chunks(self, chunks: list[Chunk], progress=None) -> None:
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self.matrix is None:
            raise RuntimeError("index_chunks() must be called first")
        q = self.vectorizer.transform([question])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[: min(top_k, len(self.chunks))]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in order]
