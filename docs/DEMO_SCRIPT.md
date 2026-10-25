# Demo Script — DLQA (8–10 minutes)

Works for the live viva and for recording the demo video. Times are cumulative.

## Before the demo (do this 10 minutes early)

```bash
cd ~/Desktop/DLQA
.venv/bin/streamlit run app/streamlit_app.py
```

- Open http://localhost:8501 in the browser, pick `kettle_manual.txt` from the sample
  dropdown and ask one throwaway question so the models are loaded and warm.
- Click **Clear chat**. Set the sample dropdown back to **(none)**.
- Have a second terminal open at the repo root for the evaluation step.
- Zoom the browser to 125 % so the back row can read it.
- Keep `data/eval/results.md` open in an editor tab.

If the laptop has no internet, that is fine: models are cached. If the UI will not start,
fall back to the CLI (see "Plan B" at the bottom).

---

## 0:00 — Opening (30 s)

> "This is a document question answering system. You give it any document, ask questions in
> plain English, and it finds the exact answer inside the document, or tells you the answer
> isn't there. Everything you'll see runs on this laptop's CPU, no cloud, no API."

Point at the sidebar's **How it works** list. Don't read it; you'll show it.

## 0:30 — Load a document (45 s)

- Pick **kettle_manual.txt** from the sample dropdown (or upload a PDF if the examiner
  hands you one; the PDF path is tested).
- Point at the header line: *"539 words, 5 chunks"*.

> "The document is split into overlapping chunks of about 180 words, because the reader
> network can only look at around 500 tokens at a time. Each chunk is turned into a 384-number
> vector by a sentence-embedding network and stored in a FAISS index."

## 1:15 — Easy factual question (45 s)

Ask: **What is the capacity of the kettle?**

Expected: **1.7 liters**, confidence ~100 %.

- Expand **Evidence passage**. Show the yellow highlight.

> "The answer isn't generated — it's a span copied out of the document. The model predicts a
> start token and an end token. That's why it can't make things up."

## 2:00 — A paraphrased question (45 s)

Ask: **Which smart speakers does it work with?**

Expected: **Amazon Alexa and Google Assistant**, confidence ~95 %.

> "Notice the words 'smart speakers' appear nowhere in the manual. The retriever matched the
> question by meaning, and the reader still found the exact span."

Backup paraphrases that are verified to work: **Which setting is best for green tea?**
(→ 60 degrees) and **How do I make a warranty claim?** (→ the contact-support sentence).

Do **not** ask about pairing mode in the demo: the reader returns the Wi-Fi troubleshooting
line instead of the KEEP WARM instruction. It is a known miss in the error analysis.

## 2:45 — A question the document can't answer (45 s)

Ask: **How much does the kettle cost?**

Expected: *"The document does not seem to contain an answer to that."*

> "This is SQuAD 2.0 behaviour. The reader has a 'no answer' score — the score of pointing
> start and end at the first token. If that beats every real span, we abstain. We measured
> this: 6 of 6 unanswerable test questions are correctly refused."

## 3:30 — Show the neural pipeline in code (90 s)

Switch to the editor, open `dlqa/reader.py`, scroll to `answer()`.

Walk through in this order, one sentence each:
1. tokenizer joins question and passage;
2. the model returns `start_logits` and `end_logits`, one number per token;
3. we add every start to every end, mask end < start and spans over 40 tokens;
4. argmax gives the span; token offsets map it back to characters;
5. `null_score = start_logits[0] + end_logits[0]`; abstain if it wins.

> "Earlier this year Hugging Face removed its ready-made QA helper, so this is implemented
> directly on the model outputs. Every line of the decision is ours."

## 5:00 — Baseline comparison (60 s)

Back in the UI, switch the sidebar radio to **TF-IDF (keyword baseline)**.

Ask: **What is the capacity of the kettle?** (works) then
**Which smart speakers does it work with?** (also works here — the document is short enough
that both retrievers surface the same chunk; say so).

> "TF-IDF is classical keyword matching with no neural network. On short documents with
> keyword-style questions it does fine — our evaluation shows that honestly. The dense
> retriever earns its place on paraphrase and on long documents, where only a few of many
> chunks are read."

Switch back to **Dense**.

## 6:00 — Evaluation (90 s)

In the second terminal:

```bash
.venv/bin/python scripts/evaluate.py --retriever dense
```

While it runs (~20 s) show `data/eval/results.md` in the editor.

> "We wrote 56 questions over three documents of different styles, with gold answers, and
> measure Exact Match and F1 the same way the SQuAD leaderboard does. Overall EM 0.86, F1 0.88,
> every gold passage retrieved in the top 5, every unanswerable question refused, about a
> fifth of a second per question."

Then one sentence on the sweep:

> "We tuned top-k and the abstain threshold on this set. Going from top-k 3 to 5 took F1 from
> 0.78 to 0.88. Loosening the abstain threshold pushed F1 to 0.93 but made the system answer
> '1.7 liters' when asked the price — so we kept the strict threshold."

## 7:30 — Limitations and future work (45 s)

> "Three honest limitations. One: no conversation memory — ask 'what does it do' after
> asking about SQuAD and it won't know 'it' means SQuAD. Two: answers are extracted, never
> summarised, so 'what is this document about' isn't answerable. Three: our test documents are
> short; a 30-page PDF is where the dense retriever would really separate from keywords.
> Next step is fine-tuning the reader on SQuAD 2.0 ourselves and measuring the gain."

## 8:15 — Close (15 s)

> "To summarise: a retriever-reader system built from two transformer networks, a chat UI
> with evidence, and a reproducible evaluation. Happy to take questions."

---

## Likely viva questions and short answers

- **Why not just send the whole document to the model?** 512-token limit; and running the
  reader on every chunk of a long document is slow. The retriever makes it O(top-k).
- **Why extractive instead of ChatGPT-style generation?** No hallucination, evidence is
  built in, small model runs on CPU, and span prediction is a clean supervised DL task.
- **What's the loss function the reader was trained with?** Cross-entropy on the start
  position plus cross-entropy on the end position (averaged). For unanswerable examples
  both targets are token 0.
- **What is the embedding model trained on?** Contrastive learning on sentence pairs:
  similar sentences are pulled together, dissimilar pushed apart (Sentence-BERT).
- **Why cosine similarity?** Vectors are normalised, so inner product equals cosine; it
  ignores vector length and compares direction, i.e. meaning.
- **What is F1 here?** Token-level overlap between predicted and gold span, harmonic mean of
  precision and recall; gives partial credit for near-misses.
- **Did you train anything?** No — Track A uses pretrained weights; the contribution is the
  system, the span-extraction implementation, and the evaluation. Fine-tuning is the planned
  next step (describe Track B).
- **How would you handle a scanned PDF?** OCR first (Tesseract), then the same pipeline.
- **How does it scale to 1,000 documents?** Same FAISS index over all chunks; add a document
  id to each chunk; the reader cost is unchanged because it still reads top-k.

---

## Plan B — CLI-only demo (if the UI fails)

```bash
.venv/bin/python -m dlqa.cli data/samples/kettle_manual.txt --show-chunks
```

Then type the same questions in the same order. `--show-chunks` prints the retrieved chunks
and their similarity scores, which actually makes the retriever step more visible.
