# DocQuery — Viva ke liye 6 parts

Project ko 6 barabar hisson mein baanta hai. Har part **akela explain ho sakta hai**, lekin sab milke
poora system banate hain. Koi bhi member koi bhi part le le; har part mein hai:
**kya cover karna hai → 2-minute mein kaise bolna hai → kaun si file dikhani hai → yaad rakhne wale numbers → teacher ke possible sawaal.**

Ek line jo har koi bole: *"DocQuery ek extractive question-answering system hai: user document upload karta hai,
sawaal poochta hai, aur system document ke andar se exact jawaab nikaal kar evidence ke saath dikhata hai;
jawaab na ho toh mana kar deta hai."*

Pipeline (sab ko yaad rahe):
```
Document → [1] Loader → [2] Chunker → [3] Embedding model (NN) → FAISS index
Question → Embedding model → top-5 chunks → [4] Reader (NN) → [5] Generator (NN, optional) → answer + evidence
```

---

## Part 1 — Problem, scope aur architecture (the "why")

**Cover karna hai:** problem statement, humne "system" kyun banaya chatbot nahi, extractive vs generative QA,
retriever + reader architecture, "yeh sirf web app nahi hai".

**2-minute script (Hinglish):**
Instructor ne title diya tha "Deep Learning-Based Document Question Answering". Problem simple hai: ek 40-page
manual ya syllabus mein ek fact dhoondhna. Search box sirf shabd match karta hai, matlab nahi; aur ChatGPT-type
chatbot document ke bahar se bhi bol deta hai, kabhi galat. Humne beech ka raasta chuna: **extractive QA**.
Jawaab generate nahi hota, document ke text ka ek **span** (tukda) hota hai, isliye hallucination possible hi nahi
aur evidence built-in hai. Architecture do neural networks ka hai: **retriever** (document ke sahi hisse dhoondhta hai)
aur **reader** (us hisse se exact jawaab nikaalta hai). Do kyun? Kyunki transformer ek baar mein ~512 tokens hi padh
sakta hai, poora document nahi; retriever pehle document ko 5 chhote tukdon tak narrow karta hai. Web page
(Streamlit, ~150 lines) sirf input-output window hai; project uske peeche ka pipeline hai: 3 neural networks,
structure-aware chunking, span extraction jo humne khud likha, abstain logic aur evaluation framework.

**Files:** `README.md` (How it works diagram), `PLAN.md` section 2, `app/streamlit_app.py` (dikhao ki UI patli hai).

**Numbers:** 6 stages, 3 neural networks (embedding, reader, generator), reader limit 512 tokens, chunk 180 words.

**Possible sawaal:**
- *Chatbot se kya alag hai?* → Chatbot text generate karta hai; hum document se span extract karte hain, source ke saath. Jawaab na ho toh "not found".
- *Deep learning kahan hai?* → Teen transformer networks: bge-base (embeddings), DeBERTa-v3 (span prediction), Flan-T5 (seq2seq). UI DL nahi hai, pipeline hai.
- *Retriever aur reader dono kyun?* → Context-length limit aur speed; reader ko poora document dena possible nahi.
- *Training ki?* → Nahi; pretrained SQuAD 2.0 models ko measurement se choose kiya (Part 6 detail deta hai).

---

## Part 2 — Loader aur Chunker (document → tukde)

**Cover karna hai:** har file type se text nikaalna, structure (heading/table/page) bachaana, OCR, chunking kyun aur
kaise, chunk size ek hyperparameter kyun hai.

**2-minute script:**
Pehla kaam file ko text banana hai, lekin structure ke saath. `loader.py` PDF (PyMuPDF, layout-aware, tables, page
numbers), Word, PowerPoint, Excel/CSV (har row ek sentence), HTML, TXT/Markdown aur images/scanned PDF (Tesseract
OCR) ko **blocks** mein badalta hai: heading block, paragraph block, table-row block, har ek pe page number.
Phir `chunker.py` blocks ko ~**180 words** ke chunks mein pack karta hai, kyunki reader model ek baar mein sirf 512
tokens padh sakta hai. Khaas baat: chunk apna **section heading saath rakhta hai** aur table ki row kabhi beech se
nahi kat-ti. Isi se baad mein citation banti hai (file · page · section). Humne pehle simple sliding window banaya
tha (180 words, 40 overlap); structure-aware karne pe syllabus ke header sawaal (instructor, office, TA) jo miss ho
rahe the sab sahi ho gaye, us document pe F1 0.86 → 0.97. Ek problem bhi aayi: har chhote section ka apna chunk
banane pe reader overconfident ho gaya (kettle ka "colour" poocha toh "blue" bol diya, ring-light wali line se).
Fix: chhote sections ko ek chunk mein pack karo. Lesson: chunk boundary bhi ek hyperparameter hai, data pe naapna padta hai.

**Files:** `dlqa/loader.py`, `dlqa/chunker.py`, `data/samples/` (3 sample docs), `requirements.txt` (pymupdf, python-docx, pytesseract).

**Numbers:** 180 words/chunk, 40-word overlap (sliding window version), 512-token reader limit, 8 file types + OCR.

**Possible sawaal:**
- *Chunk 180 hi kyun?* → `scripts/sweep.py` se chunk size × top-k × null-margin sweep kiya; 180 + top-k 5 best tha.
- *Overlap kyun?* → Taaki jawaab do chunks ki boundary pe kat na jaye.
- *Scanned PDF?* → PyMuPDF text nahi deta toh page ko image bana ke Tesseract OCR; UI warning deta hai agar text nahi mila.
- *Table rows ka kya?* → Row ko ek sentence banate hain ("Capacity | 1.7 L") aur kabhi split nahi karte.

---

## Part 3 — Retriever: embeddings aur FAISS (sahi tukda dhoondhna)

**Cover karna hai:** embedding model kya hai, vector similarity, FAISS, top-k, TF-IDF baseline, dense vs keyword ka
honest result, embedding model ka comparison.

**2-minute script:**
Pehla neural network **embedding model** hai: `BAAI/bge-base-en-v1.5` (BERT-based, sentence-transformers). Har chunk
ko **768 numbers ke vector** mein badalta hai; jin texts ka matlab same hai unke vectors paas hote hain (cosine
similarity). Sawaal ko bhi same model se vector banate hain, aur **FAISS** (Facebook AI Similarity Search) in-memory
index se sabse paas wale **top-5** chunks turant nikaal deta hai. Example: "Which smart speakers does it work with?"
— document mein "speaker" shabd hai hi nahi, phir bhi "Alexa and Google Assistant" wala chunk milta hai, kyunki
matlab match hua. Keyword search yeh nahi kar sakta. Comparison ke liye humne **TF-IDF baseline** bhi rakha hai
(`retriever.py` mein switch). Honest result: chhote test documents pe top-k 5 mein dono barabar score karte hain, kyunki
document mein sirf 4–5 chunks hi hain, sab top-5 mein aa jaate hain. Fark paraphrase sawaalon pe aur lambe documents
pe dikhta hai (50 chunks mein se 5 chunna). Embedding models bhi compare kiye: MiniLM (Recall@2 0.84) vs bge-base
(0.96), same latency, isliye bge-base default.

**Files:** `dlqa/retriever.py`, `data/eval/embeddings.md`, `scripts/compare_models.py --embeddings`.

**Numbers:** 768-dim vectors, top-k 5, Recall@5 = 1.00, bge-base 109M params, Recall@2: MiniLM 0.84 / mpnet 0.88 / bge-small 0.92 / bge-base 0.96.

**Possible sawaal:**
- *Embedding kya hota hai?* → Text ka fixed-size dense vector jisme semantic meaning encode hota hai; trained contrastively taaki similar sentences paas aayein.
- *FAISS kyun, simple loop kyun nahi?* → FAISS optimized nearest-neighbour search hai; hazaaron chunks pe bhi milliseconds.
- *top-k 3 se 5 kyun?* → Sweep mein Recall@k 0.88 → 1.00 aur F1 0.78 → 0.88.
- *Dense retriever ka fayda kab?* → Paraphrase aur long documents; isliye next step A4b: 20+ page real PDF pe eval.

---

## Part 4 — Reader: span extraction aur "no answer" (project ka dil)

**Cover karna hai:** DeBERTa reader, start/end logits, best span ka formula, null score se abstain, confidence floor,
null-margin trade-off, yeh logic humne khud likha.

**2-minute script:**
Dusra neural network **reader** hai: `deepset/deberta-v3-base-squad2`, SQuAD 2.0 pe fine-tuned. Input ek sequence
hai: `[CLS] question [SEP] chunk [SEP]`. Transformer har token ke liye ek hidden vector deta hai, aur upar ek linear
layer har token ke liye **do numbers** deta hai: **start logit** (jawaab yahan shuru) aur **end logit** (yahan
khatam). Hum har (start i, end j) pair ka score `start[i] + end[j]` jodte hain, condition: j ≥ i, span ≤ 40 tokens,
dono chunk ke andar. Sabse bada score = jawaab, jaise "two-year". Token 0 ([CLS]) ka score `start[0] + end[0]`
**"no answer" score** hai — SQuAD 2.0 mein 50k unanswerable examples se model yeh seekhta hai. Agar null score best
span se bada ho toh hum abstain karte hain: "Not found in document". Yeh poora logic `reader.py` mein humne khud likha
hai (transformers v5 ne ready-made QA pipeline hata diya tha), isliye har line explain kar sakte hain. Har retrieved
chunk pe reader chalta hai, sabse confident span jeet-ta hai. Do safety knobs: **null margin** (0 rakha; +3 karne pe
F1 0.93 lekin "How much does the kettle cost?" ka jawaab "1.7 liters" aane laga — confident galat jawaab abstain se
bura hai) aur **confidence floor 0.6** (eval mein har galat jawaab < 0.6 tha, har sahi > 0.7).

**Files:** `dlqa/reader.py` (docstring + `answer()` method), `data/eval/models.md`.

**Numbers:** DeBERTa-v3-base 184M params, 0.28 s/question CPU, max span 40 tokens, max_length 384, confidence floor 0.6, null margin 0, abstain 6/6.

**Possible sawaal:**
- *Training loss kya thi?* → Start aur end position pe cross-entropy (SQuAD fine-tuning); unanswerable ke liye target position 0.
- *DeBERTa RoBERTa se better kyun?* → Disentangled attention (content aur position alag attend) + ELECTRA-style pretraining; humare test pe F1 0.95 vs 0.89.
- *Confidence kaise nikalte ho?* → Start aur end logits pe softmax, dono probabilities ka product/joint.
- *Agar jawaab do chunks mein bata ho?* → Limitation; overlap aur structure-aware chunking isko kam karta hai.

---

## Part 5 — Generator, document overview aur product features

**Cover karna hai:** Flan-T5 ka role, grounded rule, hallucination jo humne dekha, summary/suggested questions,
multi-document citations, follow-up questions, export, UI.

**2-minute script:**
Teesra network **optional** hai: `google/flan-t5-base`, ek seq2seq transformer (encoder-decoder). Sidebar toggle
"Generative answers" on karo toh reader ka span leke poora sentence likhta hai: "The AquaBoil Pro is covered by a
two-year limited warranty." Lekin rule sakht hai: **span nahi mila toh generator bhi chup**. Kyun? Jab humne isse
passages dekar seedha jawaab dene diya, unanswerable "How much does it cost?" pe isne "1.2 kilograms without water"
bana diya — hallucination. Isliye extractive reader core hai, generator sirf finishing; aur agar generated sentence
mein span exactly nahi hai toh plain template fallback. Generator se do aur features: upload pe **summary**, aur
**suggested questions** — model 12 banata hai, hum har ek ko reader se poochte hain aur sirf confidently answerable
hi dikhate hain, isliye har chip click pe kaam karta hai. Product features: **multiple documents** ek session mein,
har jawaab pe citation (file · page · section); **follow-up**: chhota sawaal + pronoun ("where is their office?") ho
toh pichla sawaal retrieval mein jod dete hain; chat **export** Markdown mein; config env variables se; upload limit
aur OCR warnings. Generator lazy-load hota hai taaki extractive path halka rahe.

**Files:** `dlqa/generator.py`, `dlqa/pipeline.py` (`ask()`, `summarize()`, `suggest_questions()`), `app/streamlit_app.py`.

**Numbers:** Flan-T5-base ~250M params (~1 GB fp32), 12 candidate questions → filtered, context window 220 words around span.

**Possible sawaal:**
- *Yeh RAG hai?* → Haan, retrieval-augmented; lekin humara default extractive hai, generative sirf grounded rewrite.
- *Generator hallucinate kare toh?* → Guard: span zaroori + span sentence mein hona chahiye, warna template.
- *Prompt truncation issue?* → Pehle question context ke baad tha aur cut ho jaata tha; fix: context ko span ke aas-paas 220 words tak kaata.
- *Follow-up kaise?* → Simple heuristic (short + pronoun → previous question append); full conversation memory limitation hai.

---

## Part 6 — Evaluation, model selection, deployment aur limitations (the "proof")

**Cover karna hai:** test set, metrics, results, sweep se tuning, 4 readers ka comparison, training kyun nahi,
deployment (Streamlit Cloud, Docker, CI), limitations, next steps.

**2-minute script:**
Humne **56 sawaal** khud likhe, 3 alag documents pe (kettle manual, course syllabus, transformers overview), 6 aise
jinka jawaab document mein nahi. Metrics SQuAD wale: **Exact Match**, **F1** (token overlap), **Recall@k** (sahi
chunk top-k mein aaya?), **abstain accuracy**, **latency**. Result: EM 0.89, F1 0.95, Recall@5 1.00, abstain 6/6,
0.28 s/question laptop CPU pe, bina GPU, bina API. Tuning guess se nahi, measurement se: `sweep.py` ne chunk size ×
top-k × null margin try kiya; top-k 3 → 5 ne F1 0.78 → 0.88 diya. Phir `compare_models.py` se 4 pretrained readers:
roberta-base F1 0.89 (0.19 s), **deberta-v3-base 0.95 (0.28 s) ← default**, roberta-large 0.95 (0.58 s),
deberta-v3-large 0.97 (0.89 s, Docker default). **Training kyun nahi:** yeh models pehle hi SQuAD 2.0 ke 180k
examples pe fine-tuned hain; wahi base, wahi data dobara train karne se wahi result, sirf GPU time jalta. Fine-tuning
tab pay karti jab apne domain ke hazaaron Q/A pairs hote. Isliye "model selection by measurement" kiya; Track B plan
mein DistilBERT ko Colab pe fine-tune karke before/after dikhana hai. Deployment: Streamlit Community Cloud (live),
Dockerfile, GitHub Actions CI jo unit tests chalata hai. **Limitations honestly:** test set chhota aur humara likha
hua, conversation memory simple, scanned docs OCR pe depend, aur chhote docs pe dense = TF-IDF. Next: long real PDFs pe eval.

**Files:** `scripts/evaluate.py`, `scripts/sweep.py`, `scripts/compare_models.py`, `data/eval/results.md`, `models.md`, `embeddings.md`, `Dockerfile`, `.github/workflows/ci.yml`.

**Numbers:** 56 Q / 3 docs / 6 unanswerable; EM 0.89, F1 0.95, Recall@5 1.00, abstain 6/6, 0.28 s; top-k 3→5: F1 0.78→0.88; SQuAD 2.0 = 130k answerable + 50k unanswerable.

**Possible sawaal:**
- *EM aur F1 mein fark?* → EM: exact string match; F1: predicted aur gold tokens ka overlap (partial credit).
- *Test set itna chhota kyun?* → Hand-written, manually verified; limitation maante hain, next step bada real doc.
- *Known failures?* → "How do you enter pairing mode?" pe reader galat line deta hai; phrasing mismatch ("late penalty" vs "lose 10 percent per day") pe abstain. Error analysis mein likha hai.
- *Cloud pe kya issue aaya?* → ~1 GB RAM limit: base model, lazy generator, flan-t5-small fallback; file-watcher ka torchvision import error fix kiya.
