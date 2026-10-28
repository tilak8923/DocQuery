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


def _total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


# Flan-T5-base is ~1 GB in fp32; together with the reader and the embedder it does
# not fit a small host such as Streamlit Community Cloud (~2.7 GB), so fall back to
# flan-t5-small there unless DLQA_GEN_MODEL says otherwise.
_DEFAULT_GEN = "google/flan-t5-base" if _total_ram_gb() >= 6 else "google/flan-t5-small"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Chosen by scripts/compare_models.py --embeddings (data/eval/embeddings.md):
    # Recall@2 0.96 vs 0.84 for MiniLM at the same latency. bge-small (0.92) for tiny hosts.
    embed_model: str = os.environ.get("DLQA_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    # Chosen by scripts/compare_models.py (data/eval/models.md): F1 0.95 vs 0.89 for
    # roberta-base at 1.5x the latency; deberta-v3-large reaches 0.97 at 4.5x.
    reader_model: str = os.environ.get("DLQA_READER_MODEL", "deepset/deberta-v3-base-squad2")
    gen_model: str = os.environ.get("DLQA_GEN_MODEL", _DEFAULT_GEN)
    chunk_words: int = _int("DLQA_CHUNK_WORDS", 180)
    overlap_words: int = _int("DLQA_OVERLAP_WORDS", 40)
    top_k: int = _int("DLQA_TOP_K", 5)
    max_upload_mb: int = _int("DLQA_MAX_UPLOAD_MB", 50)


settings = Settings()
