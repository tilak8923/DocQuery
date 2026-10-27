# DocQuery — Deep Learning-Based Document Question Answering — Project Plan

This file holds **two tracks**. Track A is the committed scope. Track B is done only if Track A
is finished and time remains. Update the status checkboxes as work lands.

---

## 1. What the project is

A **system** with a thin UI, not a website and not a plain chatbot.

- The UI is a small layer: upload a document, ask questions, read answers with evidence.
- The core is a deep-learning pipeline that reads the document and answers questions
  **grounded only in that document**.

One-line description for the report:

> A system that lets a user upload a document (PDF/DOCX/TXT) and ask natural-language questions
> about it. A retriever neural network finds the relevant passages, and a transformer-based reader
> network extracts the answer from those passages, showing the supporting text as evidence.

---

## 2. Architecture (Retriever + Reader)

```
Document ──► Loader ──► Chunker ──► Embedding model (DL) ──► Vector index (FAISS)
                                                                   │
Question ──► Embedding model ──► top-k similar chunks ──► Reader model (DL) ──► answer + evidence
```

| Stage | What it does | Tech |
|---|---|---|
| Loader | Extract raw text from PDF/DOCX/TXT | PyMuPDF, python-docx |
| Chunker | Split text into overlapping passages (~200 words) | custom |
| Embedding model | Turn chunks and question into dense vectors | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | Similarity search over chunk vectors | FAISS (in-memory) |
| Reader model | Given question + passage, predict the answer span | `deepset/roberta-base-squad2` |
| UI | Upload, chat, show answer + source passage | Streamlit |

**Why two models?** Transformers accept ~512 tokens. A document does not fit. The retriever
narrows it to the few passages that matter; the reader answers precisely from those.

**Why extractive?** The reader predicts start/end tokens inside the passage. It cannot invent
text, it is small, it runs on CPU, and SQuAD v2 training lets it say "no answer in document".

**Runtime:** everything runs locally on a laptop CPU. Models download once from Hugging Face
into a cache folder and run offline afterwards. No cloud service is called per question.

---

## 3. Track A — Core (committed)

No training. Pretrained models only. Marks come from the system, the explanation, and the evaluation.

### Phases

| # | Phase | Deliverable | Status |
|---|---|---|---|
| A0 | Setup | Repo layout, venv, requirements, sample docs | [x] |
| A1 | Core pipeline (CLI) | `python -m dlqa.cli doc.pdf "question"` → answer + passage | [x] |
| A2 | Retriever tuning | Chunk size / overlap / top-k chosen by measurement (`scripts/sweep.py`) | [x] |
| A3 | UI | Streamlit app: upload → chat → answer with evidence + confidence | [x] |
| A4 | Evaluation | 56 hand-written Q/A on 3 docs; EM/F1/Recall@k/latency table (`data/eval/results.md`) | [x] |
| A4b | Long-document eval | Repeat A4 on one real 20+ page PDF so dense vs TF-IDF retrieval actually differ | [ ] |
| A5 | Report + demo | Outline `docs/REPORT_OUTLINE.md` and script `docs/DEMO_SCRIPT.md` drafted; report text, slides, video still to do | [~] |

### Features

Must have
- Upload PDF / DOCX / TXT
- Chat-style question answering
- Answer grounded in the document, source passage shown
- "Answer not found in document" when the model's null-answer score wins

Should have
- Conversation history per document
- Confidence score shown to user
- TF-IDF baseline retriever for comparison

### Evaluation (no training needed)
- End-to-end accuracy on the hand-made test set (exact/partial match, judged manually)
- Retriever Recall@k: is the passage containing the answer in the top-k?
- Latency per question on CPU
- Optional zero-cost comparison: `roberta-base-squad2` vs `roberta-large-squad2`

---

### Findings so far (2026-10-25)

- Defaults chosen by sweep: chunk 180 words, overlap 45, **top-k 5**, null margin 0.
- Overall on 56 questions: **EM 0.86, F1 0.88, Recall@5 1.00, abstain accuracy 1.00, 0.21 s/question** on CPU.
- Top-k 3 → 5 was the big win: it lifted dense Recall@k from 0.88 to 1.00.
- A positive null margin raises F1 (0.93 at margin 3) but makes the system answer
  unanswerable questions (e.g. "How much does the kettle cost?" → "1.7 liters"). Kept at 0.
- Dense and TF-IDF tie at top-k 5 because the sample docs have only 4–5 chunks, so the
  reader sees everything. At top-k 3 TF-IDF was slightly ahead (keyword-heavy questions).
  A long real document is needed to show the dense retriever's value → A4b.
- Remaining misses are all reader abstains on answerable questions, mostly phrasing
  mismatches ("late penalty" vs "lose 10 percent per day"). This is the strongest
  argument for Track B fine-tuning.

## 4. Track B — Stretch (only if Track A is done and time remains)

Goal: show that we can train, not just use, a deep network.

| # | Step | Deliverable | Status |
|---|---|---|---|
| B1 | Learn the HF training loop | Run the official SQuAD fine-tuning notebook once on Colab | [ ] |
| B2 | Fine-tune a small reader | `distilbert-base-uncased` on a SQuAD v2 subset, 1–2 h on Colab free GPU | [ ] |
| B3 | Evaluate | EM / F1 on SQuAD v2 dev, before vs after fine-tuning | [ ] |
| B4 | Plug in | Download the saved model folder into `models/`, point `reader.py` at it | [ ] |
| B5 | Compare | Our fine-tuned model vs the pretrained one on our own test set | [ ] |

**How Colab connects:** it does not, at runtime. Colab is only a rented GPU for the one-time
training step. The trained weights are downloaded once and loaded from disk by the app.

Fallback if B2 is too slow: fine-tune for one epoch on 5k examples, report the trend, not the peak.

---

---

## 4b. Track C — Production (started 2026-10-27)

Goal: make DocQuery usable as a real tool, not a demo. Track A stays the DL core; Track C
wraps it in the features a product needs. Order is by user-visible value.

| # | Feature | What it means | Status |
|---|---|---|---|
| C1 | Any document type | PDF (layout-aware, tables, page numbers), DOCX (headings, tables), PPTX, XLSX/CSV (rows → sentences), HTML, TXT/MD, images and scanned PDFs via OCR | [x] |
| C2 | Structure-aware chunking | Chunks follow paragraphs/headings, carry section title + page, never split a table row | [x] |
| C3 | Generative mode switch | Sidebar toggle: extractive span (default) or a Flan-T5 generated sentence grounded in the retrieved passages; both shown when on | [x] |
| C4 | Document overview | On upload: short summary + suggested questions (click to ask) | [x] |
| C5 | Multiple documents | Upload several files; answers cite file, page and section | [x] |
| C6 | Follow-up questions | Short questions with pronouns reuse the previous question for retrieval | [x] |
| C7 | Export + hygiene | Download chat as Markdown; config via env vars; OCR/no-text warnings; size limits | [x] |
| C8 | Ops | Dockerfile, GitHub Actions running unit tests, packages.txt for Tesseract on Streamlit Cloud | [x] |
| C9 | Scale | Persist indexes per document hash; background indexing; auth (later) | [ ] |

Track C findings (2026-10-27)
- Structure-aware chunking (headings carried into chunks) raised eval F1 0.88 → 0.89 and
  fixed every syllabus header question; packing small sections together was needed because
  one-section chunks made the reader overconfident.
- Minimum answer confidence set to 0.6: on the eval set every wrong answer scored < 0.6 and
  every right one > 0.7.
- Generative mode only rewrites a span the extractive reader found; letting Flan-T5 answer on
  its own fabricated answers for unanswerable questions.
- Suggested questions are filtered by actually asking them; only confidently answerable ones
  are shown.

Model choice (2026-10-27, `data/eval/models.md`): deberta-v3-base-squad2 is the default
reader (F1 0.95 vs 0.89 for roberta-base, 0.28 s/question). deberta-v3-large reaches 0.97
at 0.89 s/question and is the Docker default. No fine-tuning: the public SQuAD 2.0
checkpoints are the ceiling without in-domain training data.
Embeddings: bge-base-en-v1.5 default (Recall@2 0.96 vs 0.84 MiniLM, same latency; `data/eval/embeddings.md`).
Final eval with both: EM 0.89, F1 0.95, abstain 6/6, ~0.3 s/question.

Memory note: Flan-T5-base is ~1 GB in fp32. It is loaded lazily only when the toggle is on.
On Streamlit Community Cloud set `DLQA_GEN_MODEL=google/flan-t5-small` in secrets if the app
is killed when the toggle is first used.

## 5. Repo layout

```
DLQA/
  app/
    streamlit_app.py     # UI (Track A3)
  dlqa/
    __init__.py
    loader.py            # pdf/docx/txt → text
    chunker.py           # text → overlapping passages
    retriever.py         # embeddings + FAISS (+ TF-IDF baseline)
    reader.py            # QA model wrapper
    pipeline.py          # ties it together
    cli.py               # command-line entry point
  training/              # Track B only
    finetune_reader.py
    evaluate.py
  data/
    samples/             # sample documents
    eval/                # hand-written test questions
  models/                # fine-tuned weights (Track B), git-ignored
  PLAN.md
  README.md
  requirements.txt
```

## 6. Stack

Python 3.10+, PyTorch (CPU), `transformers`, `sentence-transformers`, `faiss-cpu`, `pymupdf`,
`python-docx`, `scikit-learn` (TF-IDF baseline), Streamlit.

## 7. Risks

- Scope creep into a full web product → UI stays minimal; the pipeline is the project.
- Long documents → retriever handles it; never feed the whole doc to the reader.
- Model download size (~500 MB for the reader) → download once, cache, commit nothing.
- Track B eats Track A time → Track B does not start until A1–A4 are checked.
