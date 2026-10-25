"""Hyper-parameter sweep for Phase A2: chunk size, top-k, null margin.

    .venv/bin/python scripts/sweep.py

Prints overall EM / F1 / Recall@k / abstain accuracy for each setting on the
hand-written eval sets. Models are loaded once and shared across settings.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dlqa.pipeline import DocumentQA  # noqa: E402
from dlqa.reader import ExtractiveReader  # noqa: E402
from dlqa.retriever import DenseRetriever, TfidfRetriever  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import EVAL_DIR, _mean, evaluate_set, load_specs  # noqa: E402

CHUNK_WORDS = [100, 180]
TOP_K = [3, 5]
NULL_MARGIN = [0.0, 1.0, 2.0, 3.0]
RETRIEVERS = ["dense", "tfidf"]


def main() -> None:
    specs = load_specs()
    reader = ExtractiveReader()
    dense = DenseRetriever()  # share the embedding model too
    tfidf = TfidfRetriever()

    print(f"{'retr':6s} {'chunk':>5s} {'k':>2s} {'margin':>6s} | {'EM':>5s} {'F1':>5s} {'R@k':>5s} {'abst':>5s} {'sec':>5s}")
    rows = []
    for retr, cw, k, m in itertools.product(RETRIEVERS, CHUNK_WORDS, TOP_K, NULL_MARGIN):
        reader.null_margin = m
        qa = DocumentQA(retriever=retr, chunk_words=cw, overlap_words=cw // 4, top_k=k, reader=reader)
        qa.retriever = dense if retr == "dense" else tfidf
        all_rows = []
        for spec in specs:
            all_rows += evaluate_set(qa, spec)["rows"]
        ans = [x for x in all_rows if x["gold"]]
        una = [x for x in all_rows if not x["gold"]]
        row = {
            "retriever": retr, "chunk_words": cw, "top_k": k, "null_margin": m,
            "em": _mean(x["em"] for x in all_rows), "f1": _mean(x["f1"] for x in all_rows),
            "recall": _mean(x["recall"] for x in ans), "abstain": _mean(x["em"] for x in una),
            "sec": _mean(x["sec"] for x in all_rows),
        }
        rows.append(row)
        print(f"{retr:6s} {cw:5d} {k:2d} {m:6.1f} | {row['em']:.2f}  {row['f1']:.2f}  {row['recall']:.2f}  {row['abstain']:.2f}  {row['sec']:.2f}")

    (EVAL_DIR / "sweep.json").write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: (r["f1"], r["abstain"]))
    print("\nBest by F1:", best)


if __name__ == "__main__":
    main()
