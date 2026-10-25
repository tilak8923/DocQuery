# Project Report Outline — Deep Learning-Based Document Question Answering

Target length: 20–30 pages including figures. Each section lists what to write and where the
material already exists in this repo. Items marked **[figure]** or **[table]** should become
a numbered figure/table in the report.

---

## Abstract (half page)

One paragraph: the problem (answering questions from a user-supplied document), the approach
(retriever + extractive reader, both transformer neural networks), the result (EM 0.86 / F1 0.88
on 56 hand-written questions, 0.2 s per question on CPU, 100 % correct abstaining), and the
contribution (a working end-to-end system with a web UI and a reproducible evaluation).

## 1. Introduction (2 pages)

- 1.1 Motivation: people have documents (manuals, syllabi, papers) and want answers, not search
  results. Reading a 30-page PDF for one fact is slow.
- 1.2 Problem statement: given a document D and a natural-language question q, return the span
  of D that answers q, or state that D does not contain the answer.
- 1.3 Objectives (bullet list, keep to 4–5):
  1. Accept PDF/DOCX/TXT and extract text.
  2. Retrieve the passages relevant to a question using neural embeddings.
  3. Extract the exact answer span with a transformer reader.
  4. Abstain when the document has no answer.
  5. Provide a chat-style UI with evidence highlighting and measure accuracy.
- 1.4 Scope and limitations (one paragraph): single document per session, extractive answers
  only (no summarising or paraphrasing), English only, no conversation memory.
- 1.5 Report organisation.

## 2. Literature Review / Background (3–4 pages)

Write this from your own understanding; the key references are listed in Section 9.

- 2.1 From keyword search to neural QA: TF-IDF, BM25, why keyword matching fails on paraphrase.
- 2.2 Recurrent models (RNN/LSTM) and their limits for long context.
- 2.3 Attention and the Transformer (Vaswani et al., 2017). Explain self-attention with the
  query/key/value picture. **[figure]** scaled dot-product attention diagram.
- 2.4 Pretraining and fine-tuning: BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019).
  Masked language modelling; why pretraining makes small-data fine-tuning possible.
- 2.5 Extractive QA and SQuAD: span prediction as two token classifiers (start, end);
  SQuAD 2.0 and the "no answer" problem (Rajpurkar et al., 2016, 2018).
- 2.6 Dense retrieval: sentence embeddings (Reimers & Gurevych, 2019), FAISS nearest-neighbour
  search, retriever–reader architecture (Chen et al., 2017, DrQA; Karpukhin et al., 2020, DPR).
- 2.7 Summary table comparing approaches. **[table]** rows: TF-IDF search, LSTM reader,
  BERT reader, retriever+reader; columns: handles paraphrase, handles long docs, can abstain.

## 3. System Design (4–5 pages)

- 3.1 Architecture overview. **[figure]** the pipeline diagram from PLAN.md §2 drawn properly:
  Document → Loader → Chunker → Embedding model → FAISS index; Question → Embedding model →
  top-k chunks → Reader → answer + evidence.
- 3.2 Why two models: the 512-token limit, the cost of running the reader on every chunk,
  retriever as a filter.
- 3.3 Document loading and chunking (`dlqa/loader.py`, `dlqa/chunker.py`): 180-word chunks
  with 45-word overlap; why overlap matters (answers on boundaries). **[figure]** overlap sketch.
- 3.4 Retriever (`dlqa/retriever.py`): all-MiniLM-L6-v2, 384-d vectors, cosine similarity via
  normalised inner product, FAISS IndexFlatIP. Also the TF-IDF baseline and why it is included.
- 3.5 Reader (`dlqa/reader.py`): roberta-base-squad2. Explain in detail — this is the core:
  - input format `<s> question </s></s> passage </s>`
  - start/end logits from a linear layer on each token's hidden state
  - best span = argmax over (start, end) with end ≥ start, length ≤ 40 tokens, inside the passage
  - null score = start_logit[0] + end_logit[0]; abstain when null > best span
  - confidence = softmax over {span score, null score}
  **[figure]** token sequence with start/end logit bars.
- 3.6 Answer selection across chunks (`dlqa/pipeline.py`): run reader on each of the top-k
  chunks, keep the best non-empty span, return its chunk as evidence.
- 3.7 User interface (`app/streamlit_app.py`): upload, chat, confidence bar, highlighted
  evidence, retriever toggle. **[figure]** screenshot of an answered question.
- 3.8 Technology stack table. **[table]** Python 3.13, PyTorch, transformers 5, sentence-transformers,
  faiss-cpu, PyMuPDF, python-docx, scikit-learn, Streamlit.

## 4. Implementation (2–3 pages)

- 4.1 Repository layout (tree from PLAN.md §5).
- 4.2 Key code walkthrough: show `ExtractiveReader.answer()` (≈40 lines) and explain each step.
  Show `DocumentQA.ask()`.
- 4.3 Running the system: CLI and UI commands from README.md.
- 4.4 Engineering decisions and problems faced:
  - transformers v5 removed the `question-answering` pipeline → implemented span extraction
    manually (turned a black box into explainable code).
  - Models downloaded once and cached; everything runs offline on CPU afterwards.
  - Streamlit caching of models so they load once per server.

## 5. Evaluation (4–5 pages)

- 5.1 Test set: 56 hand-written questions on 3 documents (syllabus, product manual, technical
  overview), 6 of them unanswerable. Describe how they were written. **[table]** counts per doc.
- 5.2 Metrics: Exact Match and F1 with SQuAD normalisation (define them with the formula);
  Recall@k for the retriever; abstain accuracy; latency.
- 5.3 Main results. **[table]** copy from `data/eval/results.md`.
- 5.4 Hyper-parameter study. **[table]** / **[figure]** from `data/eval/sweep.json`:
  - top-k 3 → 5 lifted Recall@k 0.88 → 1.00 and F1 0.78 → 0.88.
  - chunk size 100 vs 180.
  - null margin: F1 rises to 0.93 at margin 3 but abstain accuracy drops to 0.83; show the
    "1.7 liters" example of a wrong confident answer. Explain why margin 0 was kept.
- 5.5 Dense vs TF-IDF: identical at top-k 5 because the documents have only 4–5 chunks;
  at top-k 3 TF-IDF was slightly ahead. Be honest: these documents are too short to show the
  dense retriever's advantage; discuss when it would matter (paraphrased questions, long docs).
- 5.6 Error analysis. **[table]** every miss from `results.md` with a one-line cause:
  - phrasing mismatch ("late penalty" vs "lose 10 percent per day")
  - list answers ("two or three")
  - the reader preferring the null answer on long enumerations
- 5.7 Latency: ≈0.2 s per question on a laptop CPU; model load ≈5 s once.

## 6. Discussion (1–2 pages)

- What worked: extractive reader is accurate and never hallucinates; abstaining is reliable.
- What did not: no conversation memory (the "what it does" example — a follow-up question
  with a pronoun went to the wrong topic); cannot answer "what is this document about".
- Threats to validity: small, self-written test set; documents written by the same people who
  wrote the questions.

## 7. Conclusion and Future Work (1 page)

- Conclusion: objectives 1–5 met; numbers.
- Future work (ordered by value):
  1. Fine-tune the reader on SQuAD v2 and compare (Track B in PLAN.md).
  2. Evaluate on long real PDFs to separate dense vs TF-IDF retrieval.
  3. Conversation memory: prepend the previous question when the new one is short.
  4. Generative mode (Flan-T5) for summarising and multi-sentence answers.
  5. OCR for scanned PDFs; multiple documents at once.

## 8. Appendices

- A. Full question sets (`data/eval/*.json`).
- B. Full sweep table (`data/eval/sweep.json`).
- C. Source listing of `dlqa/` (≈300 lines).
- D. User manual: setup and run instructions (README.md).

## 9. References (use the formal citations)

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019.
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", arXiv 2019.
- Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text", EMNLP 2016.
- Rajpurkar, Jia, Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD", ACL 2018.
- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019.
- Chen et al., "Reading Wikipedia to Answer Open-Domain Questions", ACL 2017.
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", EMNLP 2020.
- Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs" (FAISS), IEEE TBD 2019.
- Wolf et al., "Transformers: State-of-the-Art Natural Language Processing", EMNLP 2020 (Hugging Face).
- deepset, roberta-base-squad2 model card, huggingface.co/deepset/roberta-base-squad2.

---

## Figures and tables checklist

| # | What | Source |
|---|---|---|
| Fig 1 | Pipeline architecture | draw from PLAN.md §2 |
| Fig 2 | Scaled dot-product attention | redraw from Vaswani et al. |
| Fig 3 | Chunk overlap sketch | draw |
| Fig 4 | Start/end logits over tokens | draw, or plot from `reader.py` outputs |
| Fig 5 | UI screenshot with evidence highlight | run the app |
| Fig 6 | F1 vs top-k and vs null margin | plot from `data/eval/sweep.json` |
| Tab 1 | Approach comparison | write |
| Tab 2 | Technology stack | write |
| Tab 3 | Test set composition | `data/eval/*.json` |
| Tab 4 | Main results | `data/eval/results.md` |
| Tab 5 | Sweep summary | `data/eval/sweep.json` |
| Tab 6 | Error analysis | `data/eval/results.md` misses |
