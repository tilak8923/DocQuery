# DocQuery — Deep Learning-Based Document Question Answering

Upload documents, ask questions in plain English, and get answers grounded in those
documents with the exact passage shown as evidence. Everything runs locally on CPU.

Live demo: https://ask-mydoc.streamlit.app · Plan and status: [PLAN.md](PLAN.md)

## Features

- **Any document**: PDF (layout-aware, tables, page numbers), Word, PowerPoint, Excel, CSV,
  HTML, text, Markdown, and images or scanned PDFs via OCR (Tesseract).
- **Two answer modes**: *extractive* (the exact span from the document, never invents text) or
  *generative* (a Flan-T5 sentence written from the retrieved passages, with the span shown).
- **Abstains** when the document does not contain the answer.
- **Citations**: every answer names the file, page and section it came from.
- **Multiple documents** in one session; follow-up questions reuse the previous question.
- **Document overview**: summary, section list and suggested questions that are guaranteed
  answerable.
- **Export** the chat as Markdown.

## How it works

```
Document ──► Loader ──► Structure-aware chunker ──► Embedding model ──► FAISS index
                                                                            │
Question ──► Embedding model ──► top-k chunks ──► Reader (span) ──► [Generator] ──► answer + evidence
```

1. `dlqa/loader.py` turns any supported file into blocks (headings, paragraphs, table rows)
   with page numbers. Scanned pages are OCR'd when Tesseract is installed.
2. `dlqa/chunker.py` packs blocks into ~180-word chunks that carry their section title.
3. `dlqa/retriever.py` embeds chunks with `bge-base-en-v1.5` and searches them with FAISS
   (a TF-IDF baseline is included for comparison).
4. `dlqa/reader.py` runs `deberta-v3-base-squad2` on each retrieved chunk and picks the best
   answer span from the start/end logits, or abstains when the null score wins.
5. `dlqa/generator.py` (optional) rewrites the span into a full sentence with Flan-T5, and
   writes summaries and suggested questions.
6. `dlqa/pipeline.py` ties it together: multi-document index, citations, follow-ups.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for OCR of scanned PDFs and images: install Tesseract (`brew install tesseract`
on macOS, `apt install tesseract-ocr` on Debian/Ubuntu).

The first run downloads the models (~600 MB; Flan-T5-base adds ~1 GB when generative mode
is first used) into `~/.cache/huggingface`. After that everything runs offline.

## Run

UI:

```bash
streamlit run app/streamlit_app.py
```

Command line:

```bash
python -m dlqa.cli manual.pdf "How long is the warranty?"
python -m dlqa.cli manual.pdf syllabus.docx              # several documents, interactive
python -m dlqa.cli manual.pdf -q "How long is the warranty?" --generative
python -m dlqa.cli manual.pdf --summary
```

Docker:

```bash
docker build -t docquery .
docker run -p 8501:8501 docquery
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DLQA_READER_MODEL` | `deepset/deberta-v3-base-squad2` | extractive reader (see model choice below) |
| `DLQA_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | retriever embeddings (`BAAI/bge-small-en-v1.5` for tiny hosts) |
| `DLQA_GEN_MODEL` | `google/flan-t5-base` | generative model (use `google/flan-t5-small` on small hosts) |
| `DLQA_TOP_K` | `5` | passages read per question |
| `DLQA_CHUNK_WORDS` | `180` | chunk size |
| `DLQA_MAX_UPLOAD_MB` | `50` | upload limit in the UI |

## Model choice

No training: the strongest open extractive readers are already fine-tuned on SQuAD 2.0
(130k answerable + 50k unanswerable examples), and re-training the same base on the same
data reproduces them. Training only pays off with thousands of in-domain question-answer
pairs. The choice was made on our own test set with `scripts/compare_models.py`:

| Reader | Params | EM | F1 | Abstain | s/question (CPU) |
|---|---|---|---|---|---|
| deepset/roberta-base-squad2 | 124M | 0.88 | 0.89 | 1.00 | 0.19 |
| **deepset/deberta-v3-base-squad2** (default) | 184M | 0.89 | 0.95 | 1.00 | 0.28 |
| deepset/roberta-large-squad2 | 354M | 0.89 | 0.95 | 1.00 | 0.58 |
| deepset/deberta-v3-large-squad2 | 434M | 0.93 | 0.97 | 1.00 | 0.89 |

DeBERTa-v3-base gives most of the gain at 1.5x the latency and fits small hosts. For the
best accuracy on a machine with memory to spare (local, Docker), set
`DLQA_READER_MODEL=deepset/deberta-v3-large-squad2`.

Embedding models were compared at top-k 2, where retrieval actually matters on short documents:

| Embedding model | Params | Recall@2 | F1 | s/question |
|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 23M | 0.84 | 0.76 | 0.12 |
| sentence-transformers/all-mpnet-base-v2 | 109M | 0.88 | 0.81 | 0.14 |
| BAAI/bge-small-en-v1.5 | 33M | 0.92 | 0.85 | 0.09 |
| **BAAI/bge-base-en-v1.5** (default) | 109M | 0.96 | 0.88 | 0.10 |

## Evaluate

```bash
python scripts/evaluate.py          # EM, F1, Recall@k, abstain accuracy, latency
python scripts/sweep.py             # chunk size × top-k × null margin
```

Current results on 56 hand-written questions over 3 documents (CPU): EM 0.89, F1 0.95,
Recall@5 1.00, abstain accuracy 1.00, ~0.3 s per question. See `data/eval/results.md`.

## Deploy on Streamlit Community Cloud

Push to a public GitHub repo, create the app at https://share.streamlit.io with main file
`app/streamlit_app.py`. `packages.txt` installs Tesseract there. If the app runs out of
memory when generative mode is first used, add `DLQA_GEN_MODEL = "google/flan-t5-small"`
to the app's secrets.

## Tests

```bash
python -m unittest discover tests -v
```
