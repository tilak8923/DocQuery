"""Runtime settings, overridable with environment variables.

    DLQA_EMBED_MODEL   sentence-transformers model for retrieval
    DLQA_READER_MODEL  extractive QA model (SQuAD 2 style)
    DLQA_GEN_MODEL     seq2seq model for generative answers / summaries
    DLQA_CHUNK_WORDS   target words per chunk
    DLQA_TOP_K         passages read per question
    DLQA_MAX_UPLOAD_MB upload size limit enforced by the UI
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    embed_model: str = os.environ.get("DLQA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reader_model: str = os.environ.get("DLQA_READER_MODEL", "deepset/roberta-base-squad2")
    gen_model: str = os.environ.get("DLQA_GEN_MODEL", "google/flan-t5-base")
    chunk_words: int = _int("DLQA_CHUNK_WORDS", 180)
    overlap_words: int = _int("DLQA_OVERLAP_WORDS", 40)
    top_k: int = _int("DLQA_TOP_K", 5)
    max_upload_mb: int = _int("DLQA_MAX_UPLOAD_MB", 50)


settings = Settings()
