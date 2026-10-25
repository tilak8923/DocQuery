"""Command-line entry point.

    python -m dlqa.cli data/samples/transformers.txt "Who proposed the Transformer?"
    python -m dlqa.cli data/samples/transformers.txt            # interactive
    python -m dlqa.cli doc.pdf "question" --retriever tfidf     # baseline
"""

import argparse

from .pipeline import DocumentQA


def main() -> None:
    p = argparse.ArgumentParser(description="Ask questions about a document")
    p.add_argument("document")
    p.add_argument("question", nargs="?")
    p.add_argument("--retriever", choices=["dense", "tfidf"], default="dense")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--show-chunks", action="store_true", help="print retrieved chunks")
    args = p.parse_args()

    print("Loading models (first run downloads them)...")
    qa = DocumentQA(retriever=args.retriever, top_k=args.top_k)
    n = qa.load(args.document)
    print(f"Indexed {n} chunks from {args.document}\n")

    def answer(q: str) -> None:
        r = qa.ask(q)
        if r.found:
            print(f"Answer     : {r.answer}")
            print(f"Confidence : {r.confidence:.2f}")
        else:
            print("Answer     : (not found in document)")
        print(f"Time       : {r.seconds:.2f}s")
        print(f"Evidence   : ...{_trim(r.evidence, r.answer)}...")
        if args.show_chunks:
            for rc in r.retrieved:
                print(f"  [chunk {rc.chunk.id} score={rc.score:.3f}] {rc.chunk.text[:120]}...")
        print()

    if args.question:
        answer(args.question)
        return

    print("Type a question (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q)


def _trim(evidence: str, answer: str, width: int = 300) -> str:
    if not evidence:
        return ""
    if answer and answer in evidence:
        i = evidence.index(answer)
        start = max(0, i - width // 2)
        return evidence[start : start + width]
    return evidence[:width]


if __name__ == "__main__":
    main()
