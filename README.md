# DLQA — Deep Learning-Based Document Question Answering

Upload a document, ask a question, get an answer grounded in that document with the
supporting passage shown as evidence. See [PLAN.md](PLAN.md) for scope and architecture.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads two models from Hugging Face (about 600 MB total) into
`~/.cache/huggingface`. After that everything runs offline on CPU.

## Ask from the command line

```bash
python -m dlqa.cli data/samples/transformers.txt "Who proposed the Transformer?"
```

Interactive mode:

```bash
python -m dlqa.cli data/samples/transformers.txt
```

Compare against the keyword baseline:

```bash
python -m dlqa.cli data/samples/transformers.txt "How many parameters does BERT-base have?" --retriever tfidf --show-chunks
```

## Run the UI

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. Upload a PDF / DOCX / TXT in the sidebar (or pick a
sample from `data/samples/`), ask questions in the chat, and expand **Evidence passage**
to see the highlighted span the answer was taken from. The sidebar lets you switch
between the dense neural retriever and the TF-IDF keyword baseline for comparison.

## Evaluate

```bash
.venv/bin/python scripts/evaluate.py
```

Runs the hand-written question sets in `data/eval/*.json` through the pipeline with both
retrievers and writes `data/eval/results.md` (EM, F1, Recall@k, abstain accuracy, latency).
`scripts/sweep.py` repeats this across chunk sizes, top-k and the null margin.

Current results (56 questions, 3 documents, CPU): EM 0.86, F1 0.88, Recall@5 1.00,
abstain accuracy 1.00, ~0.2 s per question.

## How it works

1. `dlqa/loader.py` extracts text from PDF / DOCX / TXT.
2. `dlqa/chunker.py` splits it into overlapping ~180-word passages.
3. `dlqa/retriever.py` embeds every passage with `all-MiniLM-L6-v2` and indexes them in FAISS.
   The question is embedded the same way and the top-k closest passages are pulled out.
4. `dlqa/reader.py` runs `roberta-base-squad2` on each passage to predict the answer span,
   or returns empty when the passage has no answer.
5. `dlqa/pipeline.py` keeps the best-scoring answer and returns it with its evidence.
