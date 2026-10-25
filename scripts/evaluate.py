"""End-to-end evaluation on the hand-written question sets in data/eval/.

    .venv/bin/python scripts/evaluate.py                # both retrievers
    .venv/bin/python scripts/evaluate.py --retriever dense --top-k 5

Metrics (per document and overall):
- EM        exact match after SQuAD-style normalisation (lowercase, strip
            punctuation and articles). For unanswerable questions, EM = 1 if
            the system abstained.
- F1        token-overlap F1 between prediction and the best gold answer.
- Recall@k  did any retrieved chunk contain a gold answer? (retriever only)
- Latency   mean seconds per question (after model load).

Writes data/eval/results.md and data/eval/results.json.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dlqa.pipeline import DocumentQA  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"


# ---------------------------------------------------------------- SQuAD metrics
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def em_score(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def best_over_golds(fn, pred: str, golds: list[str]) -> float:
    if not golds:  # unanswerable: correct iff abstained
        return float(pred == "")
    return max(fn(pred, g) for g in golds)


# ---------------------------------------------------------------- evaluation
def evaluate_set(qa: DocumentQA, spec: dict) -> dict:
    qa.load(str(ROOT / spec["document"]))
    rows = []
    for item in spec["questions"]:
        golds = item["a"]
        t0 = time.perf_counter()
        r = qa.ask(item["q"])
        dt = time.perf_counter() - t0
        retrieved_text = " ".join(normalize(rc.chunk.text) for rc in r.retrieved)
        recall = (
            float(any(normalize(g) in retrieved_text for g in golds)) if golds else None
        )
        rows.append(
            {
                "q": item["q"],
                "gold": golds,
                "pred": r.answer,
                "conf": round(r.confidence, 3),
                "em": best_over_golds(em_score, r.answer, golds),
                "f1": best_over_golds(f1_score, r.answer, golds),
                "recall": recall,
                "sec": round(dt, 3),
            }
        )
    answerable = [x for x in rows if x["gold"]]
    unanswerable = [x for x in rows if not x["gold"]]
    return {
        "document": spec["document"],
        "n": len(rows),
        "em": _mean(x["em"] for x in rows),
        "f1": _mean(x["f1"] for x in rows),
        "recall": _mean(x["recall"] for x in answerable),
        "abstain_acc": _mean(x["em"] for x in unanswerable) if unanswerable else None,
        "sec": _mean(x["sec"] for x in rows),
        "rows": rows,
    }


def load_specs() -> list[dict]:
    """Question sets are the JSON files in data/eval/ that have a 'questions' key."""
    specs = []
    for p in sorted(EVAL_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "questions" in data:
            specs.append(data)
    return specs


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["dense", "tfidf", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--null-margin", type=float, default=0.0)
    ap.add_argument("--chunk-words", type=int, default=180)
    args = ap.parse_args()

    specs = load_specs()
    retrievers = ["dense", "tfidf"] if args.retriever == "both" else [args.retriever]

    results: dict[str, list[dict]] = {}
    for name in retrievers:
        print(f"\n=== retriever: {name} (top-k={args.top_k}, null-margin={args.null_margin}, chunk={args.chunk_words}) ===")
        qa = DocumentQA(
            retriever=name,
            top_k=args.top_k,
            null_margin=args.null_margin,
            chunk_words=args.chunk_words,
            overlap_words=args.chunk_words // 4,
        )
        results[name] = []
        for spec in specs:
            res = evaluate_set(qa, spec)
            results[name].append(res)
            print(
                f"{Path(spec['document']).name:20s} n={res['n']:2d}  EM={res['em']:.2f}  "
                f"F1={res['f1']:.2f}  R@k={res['recall']:.2f}  "
                f"abstain={res['abstain_acc'] if res['abstain_acc'] is not None else '-'}  "
                f"{res['sec']:.2f}s"
            )
            for x in res["rows"]:
                flag = "ok " if x["f1"] >= 0.5 else "BAD"
                print(f"    [{flag}] {x['q']}\n          pred={x['pred']!r} gold={x['gold']}")

    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2))
    (EVAL_DIR / "results.md").write_text(render_md(results, args.top_k, args.null_margin, args.chunk_words))
    print(f"\nWrote {EVAL_DIR / 'results.md'}")


def render_md(results: dict[str, list[dict]], top_k: int, null_margin: float = 0.0, chunk_words: int = 180) -> str:
    out = [f"# Evaluation results (top-k = {top_k}, null-margin = {null_margin}, chunk = {chunk_words} words)\n"]
    out.append("| Retriever | Document | N | EM | F1 | Recall@k | Abstain acc | Latency (s) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, sets in results.items():
        for r in sets:
            ab = f"{r['abstain_acc']:.2f}" if r["abstain_acc"] is not None else "-"
            out.append(
                f"| {name} | {Path(r['document']).name} | {r['n']} | {r['em']:.2f} | {r['f1']:.2f} | "
                f"{r['recall']:.2f} | {ab} | {r['sec']:.2f} |"
            )
        all_rows = [x for r in sets for x in r["rows"]]
        ans = [x for x in all_rows if x["gold"]]
        una = [x for x in all_rows if not x["gold"]]
        out.append(
            f"| **{name}** | **overall** | {len(all_rows)} | "
            f"**{_mean(x['em'] for x in all_rows):.2f}** | **{_mean(x['f1'] for x in all_rows):.2f}** | "
            f"**{_mean(x['recall'] for x in ans):.2f}** | **{_mean(x['em'] for x in una):.2f}** | "
            f"**{_mean(x['sec'] for x in all_rows):.2f}** |"
        )
    out.append("\nEM/F1 use SQuAD normalisation. Unanswerable questions count as EM=1 when the system abstains.")
    out.append("Recall@k is computed on answerable questions only.\n")
    # failures
    out.append("## Misses (F1 < 0.5)\n")
    for name, sets in results.items():
        for r in sets:
            for x in r["rows"]:
                if x["f1"] < 0.5:
                    out.append(f"- `{name}` / {Path(r['document']).name}: **{x['q']}** → `{x['pred']}` (gold: {x['gold']})")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
