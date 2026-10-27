"""Compare reader (and optionally embedding) models on the hand-written eval sets.

    .venv/bin/python scripts/compare_models.py                 # readers
    .venv/bin/python scripts/compare_models.py --embeddings    # retrievers at top-k 2

Reports EM / F1 / abstain accuracy / latency / model size so the choice is
made on our data and our CPU budget, not on a leaderboard.
Writes data/eval/models.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import EVAL_DIR, _mean, evaluate_set, load_specs  # noqa: E402

from dlqa.pipeline import DocumentQA  # noqa: E402
from dlqa.reader import ExtractiveReader  # noqa: E402
from dlqa.retriever import DenseRetriever  # noqa: E402

READERS = [
    "deepset/roberta-base-squad2",
    "deepset/deberta-v3-base-squad2",
    "deepset/roberta-large-squad2",
    "deepset/deberta-v3-large-squad2",
]
EMBEDDERS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]


def params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def run(qa: DocumentQA, specs: list[dict]) -> dict:
    rows = []
    for spec in specs:
        rows += evaluate_set(qa, spec)["rows"]
    ans = [r for r in rows if r["gold"]]
    una = [r for r in rows if not r["gold"]]
    return {
        "em": _mean(r["em"] for r in rows),
        "f1": _mean(r["f1"] for r in rows),
        "recall": _mean(r["recall"] for r in ans),
        "abstain": _mean(r["em"] for r in una),
        "sec": _mean(r["sec"] for r in rows),
        "misses": [(r["q"], r["pred"]) for r in rows if r["f1"] < 0.5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    specs = load_specs()
    results = []

    if not args.embeddings:
        retriever_cache = DenseRetriever()
        for name in READERS:
            t0 = time.perf_counter()
            try:
                reader = ExtractiveReader(model_name=name)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: could not load ({e})")
                continue
            load_s = time.perf_counter() - t0
            qa = DocumentQA(reader=reader, top_k=args.top_k or 5)
            qa.retriever = retriever_cache
            res = run(qa, specs)
            res.update(model=name, params=params_m(reader.model), load_s=load_s)
            results.append(res)
            print(f"{name:36s} {res['params']:6.0f}M  EM={res['em']:.2f} F1={res['f1']:.2f} "
                  f"abstain={res['abstain']:.2f} {res['sec']:.2f}s/q  load {load_s:.0f}s")
            for q, pred in res["misses"]:
                print(f"      miss: {q} -> {pred!r}")
        title = "Reader models (top-k 5)"
    else:
        reader = ExtractiveReader()
        k = args.top_k or 2
        for name in EMBEDDERS:
            try:
                retr = DenseRetriever(model_name=name)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: could not load ({e})")
                continue
            qa = DocumentQA(reader=reader, top_k=k)
            qa.retriever = retr
            res = run(qa, specs)
            res.update(model=name, params=params_m(retr.model), load_s=0.0)
            results.append(res)
            print(f"{name:44s} {res['params']:5.0f}M  R@{k}={res['recall']:.2f} F1={res['f1']:.2f} {res['sec']:.2f}s/q")
        title = f"Embedding models (top-k {k})"

    md = [f"# {title}\n", "| Model | Params | EM | F1 | Recall@k | Abstain | s/question |", "|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['model']} | {r['params']:.0f}M | {r['em']:.2f} | {r['f1']:.2f} | {r['recall']:.2f} | {r['abstain']:.2f} | {r['sec']:.2f} |")
    out = EVAL_DIR / ("embeddings.md" if args.embeddings else "models.md")
    out.write_text("\n".join(md) + "\n")
    (EVAL_DIR / out.name.replace(".md", ".json")).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
