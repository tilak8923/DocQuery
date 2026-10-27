# DocQuery — Team Guide (Hinglish)

Yeh file poori team ke liye hai. Isse padh ke har member ko pata hona chahiye:
project kya hai, andar kya chal raha hai, humne kya banaya, aur instructor ke sawaalon ka
jawaab kaise dena hai. Technical shabd English mein hi rakhe hain kyunki viva mein wahi
bolne padenge.

---

## 1. Project ek line mein

> User koi bhi document upload karta hai (PDF, Word, Excel, image...), sawaal poochta hai,
> aur system us document ke andar se exact jawaab dhoondh ke deta hai, evidence ke saath.
> Agar jawaab document mein nahi hai toh saaf bolta hai "nahi hai".

Yeh **chatbot nahi hai**. Chatbot apne dimaag se bolta hai. Yeh system sirf document se bolta
hai. Isko technical naam hai **Extractive Question Answering** (span extraction).

---

## 2. Pipeline: 6 steps (yeh sabse important section hai)

```
Document ──► [1] Loader ──► [2] Chunker ──► [3] Embedding model ──► [4] FAISS index
                                                                          │
Question ──► [3] Embedding model ──► top-5 chunks ──► [5] Reader ──► [6] Generator (optional)
                                                                          │
                                                            answer + evidence + page/section
```

### Step 1 — Loader (`dlqa/loader.py`)
File ko text mein badalta hai. Yahan koi deep learning nahi hai, yeh plumbing hai.
- PDF: PyMuPDF se text blocks, tables, page numbers.
- Word/PowerPoint/Excel/CSV/HTML: har format ka apna parser.
- Scanned PDF ya image: OCR (Tesseract) se text nikalta hai.
- Output: **blocks** — heading, paragraph, table row — har block ke saath page number.

Kyun important: resume ya manual jaise documents mein structure hoti hai. Agar sab ek
lambi string bana do toh "Education" heading aur degree line ka rishta toot jaata hai.

### Step 2 — Chunker (`dlqa/chunker.py`)
Text ko ~180 shabdon ke tukdon (chunks) mein kaat-ta hai.
- **Kyun kaatna padta hai**: Reader model ek baar mein sirf ~512 tokens padh sakta hai.
  Poora document nahi jaa sakta.
- Har chunk ke upar uska section heading lagta hai ("Warranty" + uska paragraph).
- Table ki row kabhi beech se nahi kat-ti.
- Har chunk yaad rakhta hai: kaunsi file, kaunsa page, kaunsa section. Isi se citation banti hai.

### Step 3 — Embedding model (`dlqa/retriever.py`, model: all-MiniLM-L6-v2)
**Pehla neural network.** Har chunk ko 384 numbers ke vector mein badalta hai. Sawaal ko bhi.
- Jin sentences ka **matlab** same hai, unke vectors paas hote hain (cosine similarity zyada).
- Isliye "smart speakers" poochne pe "Alexa and Google Assistant" wala chunk milta hai,
  jabki "speaker" shabd document mein hai hi nahi. Keyword search yeh nahi kar sakta.
- Yeh model Sentence-BERT tareeke se train hua hai: similar sentence pairs ko paas, alag
  ko door (contrastive learning).

### Step 4 — FAISS index
Facebook ki library jo vectors ko store karti hai aur sabse paas wale 5 vectors turant
dhoondh deti hai (nearest neighbour search). Hum normalised vectors ka inner product lete
hain, jo cosine similarity ke barabar hai.

**TF-IDF baseline**: sidebar mein "Advanced" ke andar ek option hai. Yeh purana keyword
tareeka hai, bina neural network ke. Humne isko isliye rakha hai taaki dikha sakein ki
neural retriever kahan behtar hai (paraphrase pe). Evaluation mein dono compare kiye hain.

### Step 5 — Reader (`dlqa/reader.py`, model: roberta-base-squad2)
**Dusra neural network, project ka dil.** Sawaal + ek chunk andar jaata hai, exact jawaab
ka span bahar aata hai. Andar kya hota hai:

1. Tokenizer sawaal aur chunk ko jodta hai: `<s> question </s></s> passage </s>`
2. RoBERTa (transformer) har token ke liye ek hidden vector banata hai.
3. Upar ek linear layer har token ke liye **do numbers** deta hai: **start logit** aur
   **end logit**. Start logit zyada = "jawaab yahan se shuru hota hai".
4. Hum har (start, end) pair ka score = start_logit + end_logit nikaalte hain, sirf unke liye
   jahan end ≥ start, span ≤ 40 tokens, aur dono passage ke andar hain. Sabse bada score = jawaab.
5. **Null score** = start_logit[0] + end_logit[0] (pehle `<s>` token pe). Agar null score best
   span se bada hai, toh model keh raha hai "is passage mein jawaab nahi hai". Tab hum
   abstain karte hain.
6. Confidence = softmax over {span score, null score}.

Top-5 chunks mein se har ek pe reader chalta hai, sabse zyada score wala non-empty span
final jawaab hota hai, aur uska chunk evidence ban jaata hai.

Yeh poora logic humne **khud likha hai** (library ka ready-made function use nahi kiya),
isliye har line explain kar sakte hain.

### Step 6 — Generator (`dlqa/generator.py`, model: Flan-T5-base) — optional toggle
**Teesra neural network.** Sidebar mein "Generative answers" on karne pe chalta hai.
- Reader ka span leta hai aur usse ek poora sentence banata hai:
  span "two-year" → "The AquaBoil Pro is covered by a two-year limited warranty."
- Sirf tab chalta hai jab reader ne span dhoondha ho. Agar reader ne "nahi mila" bola, toh
  generator bhi nahi bolta. Yeh rule humne isliye banaya kyunki free generation mein model
  ne kettle ka price "1.2 kilograms" bata diya tha (hallucination).
- Summary aur suggested questions bhi yahi model likhta hai. Suggested questions ko hum
  pehle khud reader se check karte hain; sirf wahi dikhate hain jinka jawaab document mein hai.

---

## 3. Background concepts (viva ke liye)

### Transformer aur attention
- 2017, Google, paper "Attention Is All You Need" (Vaswani et al.).
- RNN/LSTM tokens ek-ek karke padhte the; Transformer saare tokens ek saath dekhta hai.
- **Self-attention**: har token ke liye teen vectors: Query, Key, Value.
  Score = softmax(Q·Kᵀ / √d). Output = scores × V ka weighted sum.
  Matlab: har shabd baaki har shabd ko dekh ke apna matlab tay karta hai.
- Multi-head: 8–12 attention heads parallel, har ek alag rishta seekhta hai.
- Position ka pata nahi hota, isliye positional encoding jodte hain.

### BERT → RoBERTa
- BERT (2018): sirf encoder. Pretraining: masked language modeling (15% tokens chhupa ke
  predict karo) + next sentence prediction. 12 layers, 110M parameters.
- RoBERTa (2019, Facebook): BERT ka better-trained version. 10× data, bada batch, NSP hataya.
- Pretrain ek baar bada data pe, phir **fine-tune** chhote task pe (yahan: SQuAD).

### SQuAD
- Stanford ka dataset: Wikipedia passages pe 100k+ sawaal, jawaab passage ka span hai.
- SQuAD 2.0 ne 50k **unanswerable** sawaal jode. Isliye model "nahi pata" bolna seekhta hai.
- Metrics: **Exact Match** (prediction == gold, normalise karke) aur **F1** (token overlap,
  partial credit).
- Fine-tuning loss: start position pe cross-entropy + end position pe cross-entropy, average.
  Unanswerable ke liye dono target = token 0.

### Extractive vs Generative
- Extractive: jawaab document ka tukda hai. Invent nahi kar sakta. Evidence built-in.
- Generative: model naya text likhta hai. Fluent, lekin galat bhi bol sakta hai confidently.
- Humara default extractive hai, generative optional aur grounded (span ke upar).

---

## 4. Humne kya kiya (yeh confidently bolo)

1. **Poora pipeline khud banaya**: loader, structure-aware chunker, retriever, reader ka
   span-extraction logic, abstain logic, multi-document citations, follow-up handling.
2. **Koi bhi document**: PDF (tables, pages), Word, PowerPoint, Excel, CSV, HTML, text,
   images/scans via OCR.
3. **Do modes**: extractive (exact span) aur generative (grounded sentence) — toggle se.
4. **Evaluation framework**: 56 hand-written questions, 3 documents, 6 unanswerable.
   Script EM, F1, Recall@k, abstain accuracy, latency nikaalta hai.
5. **Tuning measurement se ki**: top-k 3→5 se F1 0.78→0.88; abstain threshold loose
   karne pe F1 0.93 lekin system price ka jawaab "1.7 liters" dene laga, isliye strict rakha;
   confidence floor 0.6 (galat jawaab sab < 0.6 the, sahi sab > 0.7).
6. **Deployed**: GitHub pe code, Streamlit Cloud pe live app, Docker, CI tests.

### Final numbers (yaad kar lo)

| Metric | Value |
|---|---|
| Questions | 56 (3 docs, 6 unanswerable) |
| Exact Match | 0.88 |
| F1 | 0.89 |
| Recall@5 (sahi passage top-5 mein) | 1.00 |
| Abstain accuracy | 6/6 = 1.00 |
| Speed | ~0.2 s / question, laptop CPU |

---

## 5. "Yeh toh normal web app hai" — jawaab

Pehli line: **"Web page sirf input-output window hai. Project uske peeche teen neural
networks ka pipeline hai."**

Phir dikhao, bolo mat:
1. "How much does the kettle cost?" poocho → system mana karta hai. Normal app yeh nahi karta.
2. "Which smart speakers does it work with?" poocho → "speaker" shabd document mein nahi hai,
   phir bhi sahi jawaab. Keyword search yeh nahi kar sakta.
3. Evidence expand karo → highlighted span, file, page, section.
4. `dlqa/reader.py` kholo → start/end logits, null score, argmax. Yeh humara code hai.

Unique kya hai:
- Jawaab kabhi invent nahi hota (hallucination-free by design).
- Kisi bhi document type pe chalta hai, structure ke saath.
- Laptop CPU pe offline, koi API cost nahi.
- Numbers ke saath evaluate kiya, guess se nahi.

---

## 6. Sawaal-jawaab bank

**Q: Poora document model ko kyun nahi dete?**
Transformer ~512 tokens hi le sakta hai, aur har chunk pe reader chalana slow hai. Retriever
pehle 5 relevant chunks chunta hai, reader sirf unpe chalta hai.

**Q: Training kyun nahi ki?**
Model pehle hi SQuAD 2.0 ke 180k examples pe fine-tuned hai. Wahi data pe dobara train karne
se wahi result aata. Training tab useful hai jab apne domain ke hazaaron Q/A pairs hon, jo
humare paas nahi hain. Humne isske bajaye alag-alag pretrained readers ko apne test set pe
compare kiya (roberta-base vs large vs deberta).

**Q: ChatGPT se kya alag hai?**
ChatGPT document ke bahar se bhi bol sakta hai aur galat bhi bol sakta hai confidently. Yeh
sirf document se bolta hai, evidence dikhata hai, aur nahi pata toh "nahi pata" bolta hai.

**Q: Embedding kya hoti hai?**
Text ka numerical vector jisme matlab encode hota hai. Same matlab = paas vectors.

**Q: Cosine similarity kyun?**
Vector ki length ignore karke sirf direction (matlab) compare karta hai. Normalised vectors
ka dot product = cosine.

**Q: F1 kya hai yahan?**
Predicted span aur gold answer ke tokens ka overlap: precision aur recall ka harmonic mean.
Near-miss ko partial credit milta hai.

**Q: Loss function?**
Cross-entropy on start index + cross-entropy on end index, average.

**Q: Abstain kaise hota hai?**
Null score (token 0 pe start+end logit) vs best span score. Null jeeta toh "not found".
Upar se confidence < 0.6 ho toh bhi "not found".

**Q: Chunk size 180 kyun?**
Sweep kiya: 100 vs 180 words, top-k 3 vs 5. 180 + top-k 5 pe Recall 1.00 aur best F1.

**Q: Dense aur TF-IDF mein same score kyun aaya?**
Humare test documents chhote hain (4–5 chunks); top-5 mein sab chunk aa jaate hain, toh
retriever matter nahi karta. Top-3 pe fark dikhta hai. Lambe documents pe dense retriever
ka fayda hota hai. Yeh humne honestly report kiya hai.

**Q: Limitations?**
Conversation memory simple hai (sirf previous question). Extractive hai, toh "document kis
baare mein hai" type summary sawaal span se nahi hote (uske liye generator/overview hai).
Scanned PDFs pe OCR quality pe depend karta hai. Test set chhota aur self-written hai.

**Q: Scale kaise karoge?**
FAISS index mein hazaaron documents ke chunks aa sakte hain; reader ka cost same rehta hai
kyunki woh sirf top-k padhta hai.

---

## 7. Team roles (5 log) — kaun kya bolega

| Member | Part | Kya bolna/dikhana hai |
|---|---|---|
| 1 (lead) | Problem + architecture | Section 1, 2 ka overview diagram, "normal web app" wala jawaab |
| 2 | Loader + chunker | Document types, blocks, chunk size kyun, section/page citations |
| 3 | Retriever | Embeddings, cosine, FAISS, dense vs TF-IDF demo |
| 4 | Reader | Span extraction, start/end logits, null score, `reader.py` live dikhana |
| 5 | Evaluation + generator | Numbers table, sweep findings, generative toggle demo, limitations |

Har member apna section + Q&A bank ke related sawaal ratt le. Demo script
`docs/DEMO_SCRIPT.md` mein minute-by-minute flow hai.

---

## 8. Demo se pehle checklist

- App warm karo (ek sawaal pooch lo) taaki models load ho jaayein.
- Kettle manual sample select karo.
- Sawaal order: capacity → smart speakers → price (abstain) → evidence expand → reader.py.
- Pairing mode wala sawaal **mat poocho** (known miss).
- Generative toggle tab on karo jab span wale sawaal dikha chuke ho.
