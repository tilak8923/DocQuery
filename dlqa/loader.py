"""Document loader: turns a PDF / DOCX / TXT file into plain text.

This is the only stage with no deep learning in it. Its job is just to
give the rest of the pipeline clean text to work with.
"""

from pathlib import Path


def load_document(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix == ".docx":
        text = _load_docx(path)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return _clean(text)


def load_bytes(data: bytes, filename: str) -> str:
    """Same as load_document but for an in-memory upload (used by the UI)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import fitz  # PyMuPDF

        with fitz.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    elif suffix == ".docx":
        import io

        import docx

        document = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in document.paragraphs)
    elif suffix in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return _clean(text)


def _load_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _load_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _clean(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks.
    lines = [" ".join(line.split()) for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()
