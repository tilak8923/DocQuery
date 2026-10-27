"""Command-line entry point.

    python -m dlqa.cli manual.pdf "How long is the warranty?"
    python -m dlqa.cli manual.pdf syllabus.docx            # several docs, interactive
    python -m dlqa.cli manual.pdf -q "..." --generative    # Flan-T5 sentence too
    python -m dlqa.cli manual.pdf --summary                # overview + suggested questions
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .loader import load_document
from .pipeline import DocumentQA, QAResult


def main() -> None:
    p = argparse.ArgumentParser(description="Ask questions about one or more documents")
    p.add_argument("paths", nargs="+", help="document files; a trailing non-file is taken as the question")
    p.add_argument("-q", "--question")
    p.add_argument("--retriever", choices=["dense", "tfidf"], default="dense")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--generative", action="store_true", help="also produce a generated sentence")
    p.add_argument("--summary", action="store_true", help="print a summary and suggested questions")
    p.add_argument("--show-chunks", action="store_true", help="print retrieved chunks")
    args = p.parse_args()

    paths = list(args.paths)
    question = args.question
    if question is None and len(paths) > 1 and not Path(paths[-1]).exists():
        question = paths.pop()

    print("Loading models (first run downloads them)...")
    qa = DocumentQA(retriever=args.retriever, top_k=args.top_k)
    for path in paths:
        doc = load_document(path)
        n = qa.add_document(doc)
        print(f"Indexed {doc.name}: {doc.words} words, {doc.pages or '?'} pages, {n} chunks")
    print()

    if args.summary:
        print("Summary:", qa.summarize())
        for q in qa.suggest_questions():
            print("  ?", q)
        print()

    previous: str | None = None

    def answer(q: str) -> None:
        nonlocal previous
        r = qa.ask(q, generative=args.generative, previous_question=previous)
        previous = q
        _print(r, args.show_chunks)

    if question:
        answer(question)
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


def _print(r: QAResult, show_chunks: bool) -> None:
    if r.answer:
        print(f"Answer     : {r.answer}")
        print(f"Confidence : {r.confidence:.2f}")
    else:
        print("Answer     : (not found in document)")
    if r.generated is not None:
        print(f"Generated  : {r.generated or '(model found no answer)'}")
    if r.citation:
        print(f"Source     : {r.citation}")
    print(f"Time       : {r.seconds:.2f}s")
    print(f"Evidence   : ...{_trim(r.evidence, r.answer)}...")
    if show_chunks:
        for rc in r.retrieved:
            print(f"  [chunk {rc.chunk.id} score={rc.score:.3f} p{rc.chunk.page}] {rc.chunk.text[:110]}...")
    print()


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
