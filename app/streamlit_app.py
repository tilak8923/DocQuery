"""Streamlit UI for DocQuery (DLQA).

Run:  .venv/bin/streamlit run app/streamlit_app.py

Layout
- Sidebar: upload a document, choose retriever / top-k, see index stats.
- Main: chat. Each answer shows confidence and the evidence passage with the
  answer highlighted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `dlqa` importable when run as `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlqa.pipeline import DocumentQA, QAResult  # noqa: E402

st.set_page_config(page_title="DocQuery", page_icon="📄", layout="wide")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


class SampleFile:
    """Minimal stand-in for an UploadedFile so sample docs go through the same path."""

    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading neural models (first time only)...")
def get_pipeline(retriever: str, top_k: int) -> DocumentQA:
    return DocumentQA(retriever=retriever, top_k=top_k)


def highlight(evidence: str, answer: str) -> str:
    """Return evidence as markdown with the answer span bolded/highlighted."""
    if answer and answer in evidence:
        i = evidence.index(answer)
        before, after = evidence[:i], evidence[i + len(answer) :]
        return f"{before}<mark><b>{answer}</b></mark>{after}"
    return evidence


def render_result(r: QAResult) -> None:
    if r.found:
        st.markdown(f"**{r.answer}**")
        st.progress(min(1.0, r.confidence), text=f"Confidence {r.confidence:.0%} · {r.seconds:.2f}s")
    else:
        st.markdown("_The document does not seem to contain an answer to that._")
        st.caption(f"{r.seconds:.2f}s")
    with st.expander("Evidence passage", expanded=r.found):
        st.markdown(highlight(r.evidence, r.answer), unsafe_allow_html=True)
        if r.retrieved:
            st.caption(
                "Retrieved chunks: "
                + ", ".join(f"#{rc.chunk.id} ({rc.score:.2f})" for rc in r.retrieved)
            )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 DocQuery")
    st.caption("Deep Learning-Based Document Question Answering")

    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md"])

    samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix in {".pdf", ".docx", ".txt", ".md"})
    sample_choice = st.selectbox(
        "…or use a sample document",
        ["(none)"] + [p.name for p in samples],
        disabled=uploaded is not None,
    )
    if uploaded is None and sample_choice != "(none)":
        uploaded = SampleFile(SAMPLES_DIR / sample_choice)

    st.divider()
    retriever = st.radio(
        "Retriever",
        ["dense", "tfidf"],
        format_func=lambda x: "Dense (neural embeddings)" if x == "dense" else "TF-IDF (keyword baseline)",
    )
    top_k = st.slider("Passages to read (top-k)", 1, 8, 5)

    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Document → text → overlapping chunks\n"
        "2. Chunks + question → embeddings (MiniLM)\n"
        "3. FAISS finds the closest chunks\n"
        "4. RoBERTa (SQuAD2) extracts the answer span\n"
        "5. Or abstains when no span beats the null score"
    )

# ---------------------------------------------------------------- state
qa = get_pipeline(retriever, top_k)
qa.top_k = top_k

if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-index when the document OR the retriever changes. The pipeline object is
# cached per retriever, so switching retrievers hands us a fresh, empty index.
index_key = (uploaded.name, uploaded.size, retriever) if uploaded else None
if uploaded and st.session_state.get("index_key") != index_key:
    new_doc = st.session_state.get("index_key", (None,))[0] != uploaded.name
    with st.spinner(f"Reading and indexing {uploaded.name}..."):
        try:
            qa.load_bytes(uploaded.getvalue(), uploaded.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read document: {e}")
            st.stop()
    st.session_state.index_key = index_key
    if new_doc:
        st.session_state.messages = []

# ---------------------------------------------------------------- main
st.header("Ask your document")

if not uploaded:
    st.info("Upload a PDF, DOCX or TXT in the sidebar to begin.")
    st.stop()

words = len(qa.text.split())
st.caption(f"**{uploaded.name}** · {words:,} words · {len(qa.chunks)} chunks · retriever: {retriever}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_result(m["content"])

question = st.chat_input("Ask a question about the document")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Reading..."):
            result = qa.ask(question)
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
