"""Streamlit UI for DocQuery.

Run:  .venv/bin/streamlit run app/streamlit_app.py

Sidebar: upload one or more documents (PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT,
MD, images), choose extractive or generative answers, retriever, top-k.
Main: document overview (summary + suggested questions), chat with answers,
confidence, citation (file · page · section) and the highlighted evidence.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlqa.config import settings  # noqa: E402
from dlqa.loader import SUPPORTED_TYPES, load_bytes  # noqa: E402
from dlqa.pipeline import DocumentQA, QAResult  # noqa: E402

st.set_page_config(page_title="DocQuery", page_icon="📄", layout="wide")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


class SampleFile:
    """Stand-in for an UploadedFile so sample docs go through the same path."""

    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading neural models (first time only)...")
def get_pipeline(retriever: str) -> DocumentQA:
    return DocumentQA(retriever=retriever)


def highlight(evidence: str, answer: str) -> str:
    if answer and answer in evidence:
        i = evidence.index(answer)
        return f"{evidence[:i]}<mark><b>{answer}</b></mark>{evidence[i + len(answer):]}"
    return evidence


def render_result(r: QAResult) -> None:
    if r.generated:
        st.markdown(r.generated)
        if r.answer:
            st.caption(f"Exact span: **{r.answer}**")
    elif r.answer:
        st.markdown(f"**{r.answer}**")
    else:
        st.markdown("_The document does not seem to contain an answer to that._")

    meta = []
    if r.answer:
        meta.append(f"Confidence {r.confidence:.0%}")
    if r.citation:
        meta.append(r.citation)
    meta.append(f"{r.seconds:.2f}s")
    st.caption(" · ".join(meta))
    if r.answer:
        st.progress(min(1.0, r.confidence))

    with st.expander("Evidence passage", expanded=bool(r.answer)):
        st.markdown(highlight(r.evidence, r.answer).replace("\n", "<br>"), unsafe_allow_html=True)
        if r.retrieved:
            st.caption(
                "Retrieved: "
                + ", ".join(
                    f"{rc.chunk.doc.split('.')[0][:18]} p{rc.chunk.page or '-'} ({rc.score:.2f})"
                    for rc in r.retrieved
                )
            )


def export_markdown(messages: list[dict], doc_names: list[str]) -> str:
    lines = [f"# DocQuery chat — {datetime.now():%Y-%m-%d %H:%M}", "", f"Documents: {', '.join(doc_names)}", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"**Q:** {m['content']}", ""]
        else:
            r: QAResult = m["content"]
            lines += [f"**A:** {r.generated or r.answer or '(not found in document)'}"]
            if r.citation:
                lines += [f"*Source: {r.citation}*"]
            lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 DocQuery")
    st.caption("Ask questions, get answers grounded in your documents.")

    uploads = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help=f"Up to {settings.max_upload_mb} MB each. Scanned PDFs and images are OCR'd when Tesseract is installed.",
    )
    files = [u for u in (uploads or []) if u.size <= settings.max_upload_mb * 1024 * 1024]
    for u in uploads or []:
        if u.size > settings.max_upload_mb * 1024 * 1024:
            st.warning(f"{u.name} is larger than {settings.max_upload_mb} MB and was skipped.")

    samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lstrip(".").lower() in SUPPORTED_TYPES)
    sample_choice = st.multiselect("…or add sample documents", [p.name for p in samples])
    files += [SampleFile(SAMPLES_DIR / n) for n in sample_choice]

    st.divider()
    generative = st.toggle(
        "Generative answers",
        value=False,
        help="Off: the exact span from the document (never invents text). "
        "On: a Flan-T5 model writes a full sentence from the retrieved passages. "
        "The exact span is still shown underneath.",
    )
    with st.expander("Advanced"):
        retriever = st.radio(
            "Retriever",
            ["dense", "tfidf"],
            format_func=lambda x: "Dense (neural embeddings)" if x == "dense" else "TF-IDF (keyword baseline)",
        )
        top_k = st.slider("Passages to read (top-k)", 1, 8, settings.top_k)

    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

# ---------------------------------------------------------------- state / indexing
qa = get_pipeline(retriever)
qa.top_k = top_k
st.session_state.setdefault("messages", [])

index_key = (tuple(sorted((f.name, f.size) for f in files)), retriever)
if files and st.session_state.get("index_key") != index_key:
    prev_names = set(st.session_state.get("index_key", ((),))[0])
    qa.clear()
    errors = []
    with st.spinner("Reading and indexing..."):
        for f in files:
            try:
                qa.add_document(load_bytes(f.getvalue(), f.name))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{f.name}: {e}")
    for err in errors:
        st.sidebar.error(err)
    for d in qa.documents:
        for note in d.notes:
            st.sidebar.warning(note)
    st.session_state.index_key = index_key
    st.session_state.pop("overview", None)
    if set(index_key[0]) - prev_names:  # a new document arrived -> fresh chat
        st.session_state.messages = []
elif not files and qa.documents:
    qa.clear()
    st.session_state.pop("index_key", None)
    st.session_state.pop("overview", None)

# ---------------------------------------------------------------- main
st.header("Ask your documents")

if not qa.documents:
    st.info("Upload a document in the sidebar to begin. PDF, Word, PowerPoint, Excel, CSV, HTML, text, Markdown and images are supported.")
    st.stop()

doc_names = [d.name for d in qa.documents]
st.caption(
    " · ".join(f"**{d.name}** ({d.words:,} words, {d.pages or '?'} pages)" for d in qa.documents)
    + f" · {len(qa.chunks)} chunks · {'generative' if generative else 'extractive'} mode"
)

# Overview (needs the generator; only when the toggle is on so the extractive path stays light)
if generative:
    if "overview" not in st.session_state:
        with st.spinner("Writing an overview..."):
            try:
                st.session_state.overview = {
                    "summary": qa.summarize(),
                    "questions": qa.suggest_questions(n=4),
                }
            except Exception as e:  # noqa: BLE001
                st.session_state.overview = {"summary": "", "questions": [], "error": str(e)}
    ov = st.session_state.overview
    with st.expander("Document overview", expanded=not st.session_state.messages):
        if ov.get("error"):
            st.warning(f"Overview unavailable: {ov['error']}")
        if ov.get("summary"):
            st.markdown(ov["summary"])
        if ov.get("questions"):
            st.caption("Try asking:")
            cols = st.columns(2)
            for i, q in enumerate(ov["questions"]):
                if cols[i % 2].button(q, use_container_width=True, key=f"suggest_{i}"):
                    st.session_state.pending_question = q

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_result(m["content"])

question = st.chat_input("Ask a question about the documents") or st.session_state.pop("pending_question", None)
if question:
    previous = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Reading..."):
            try:
                result = qa.ask(question, generative=generative, previous_question=previous)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not answer: {e}")
                st.stop()
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "content": result})

if st.session_state.messages:
    st.download_button(
        "Download chat as Markdown",
        export_markdown(st.session_state.messages, doc_names),
        file_name="docquery-chat.md",
        mime="text/markdown",
    )
