# DocQuery — Animated Explainer Video Brief

Yeh file video banane ke liye hai: AI video tool mein paste karo ya animator ko do.
Part A seedha paste karne wala prompt hai. Part B scene-by-scene script hai. Part C woh
context hai jo project ki code files mein kahin nahi likha (decisions, failures, lessons).
Part D paanch members ke 90-second bolne wale scripts hain.

Target: 7–8 minute video. Narration Hinglish, on-screen text English. Style: clean 2D motion
graphics, dark background, ek accent colour (orange) jo "answer span" ko highlight kare,
har neural network ek glowing box.

---

## Part A — Master prompt (copy-paste)

```
Create a 7-minute 2D motion-graphics explainer video for a college deep-learning project
called "DocQuery — Deep Learning-Based Document Question Answering".

STYLE: dark navy background, white text, one orange accent colour used only for the
"answer span" and for the three neural-network boxes. Clean flat icons, smooth easing,
no stock footage, no emojis. Narration in Hinglish (Hindi-English mix, Roman script);
all on-screen text in English. Calm narrator, not salesy.

WHAT THE PRODUCT DOES: user uploads any document (PDF, Word, PowerPoint, Excel, image),
asks a question in plain English, and the system finds the exact answer inside the
document, shows the evidence passage with the answer highlighted, plus file name, page
and section. If the answer is not in the document it says so instead of guessing.

THE PIPELINE (animate as a left-to-right rail with a moving dot):
1 Loader — file → structured blocks (headings, paragraphs, table rows, page numbers). No neural net.
2 Chunker — blocks → ~180-word chunks that keep their section heading. No neural net.
3 Embedding model (NEURAL NET #1, MiniLM) — every chunk and the question → 384-number vector; similar meaning = nearby vectors.
4 FAISS index — nearest-neighbour search, returns top-5 chunks.
5 Reader (NEURAL NET #2, DeBERTa fine-tuned on SQuAD 2.0) — for each chunk, predicts a START logit and an END logit per token; best (start,end) pair = answer span; token 0 score = "no answer".
6 Generator (NEURAL NET #3, Flan-T5, optional toggle) — rewrites the span into a full sentence; only runs when the reader found a span.

KEY SCENES (see script): the span-extraction animation with start/end bars over tokens;
the "no answer" moment where the null score wins; the paraphrase moment ("smart speakers"
finds "Alexa and Google Assistant" though the word speaker is not in the document);
the results table; the "it is not just a web app" rebuttal; the decision log.

RESULTS TO SHOW: 56 hand-written questions over 3 documents, 6 unanswerable.
Exact Match 0.89, F1 0.95, Recall@5 1.00, abstain accuracy 6/6, 0.28 s per question on a
laptop CPU, no GPU, no API, fully offline.

TONE OF THE ENDING: honest. State limitations (short self-written test set, simple
conversation memory, OCR quality dependent) and the next step (evaluate on long real PDFs).
```

---

## Part B — Scene-by-scene script

Format: **Visual** (kya dikhana hai) · **Screen text** (English) · **Narration** (Hinglish).
Timing approximate.

### Scene 1 — Problem (0:00–0:35)
- Visual: ek student 40-page PDF scroll kar raha hai, ek sawaal ke liye. Clock tick.
- Screen text: "One fact. Forty pages."
- Narration: "Manual ho, syllabus ho, research paper ho. Sawaal ek hota hai, jawaab kahin
  andar chhupa hota hai. Search box sirf shabd dhoondhta hai, matlab nahi. Aur chatbot
  document ke bahar se bhi bol deta hai, kabhi galat bhi. Humne beech ka raasta banaya."

### Scene 2 — What DocQuery does (0:35–1:10)
- Visual: UI mockup. Upload → question → answer card: bold answer, confidence bar,
  evidence passage with orange highlight, "kettle_manual.pdf · page 1 · Warranty".
- Screen text: "Ask your document. Get the exact answer, with proof."
- Narration: "Document upload karo, sawaal poocho. System document ke andar se exact jawaab
  nikaalta hai, highlight karke dikhata hai, file aur page number ke saath. Aur agar jawaab
  document mein nahi hai, toh saaf bolta hai: nahi hai."

### Scene 3 — Not a chatbot (1:10–1:35)
- Visual: do boxes. Left "Chatbot": speech bubble se text generate ho raha, source nahi.
  Right "DocQuery": document se ek tukda cut hoke bahar aa raha hai.
- Screen text: "Extractive QA: the answer is a span of the document."
- Narration: "Technical naam hai Extractive Question Answering. Jawaab generate nahi hota,
  document se extract hota hai. Isliye yeh kabhi apni taraf se kuch invent nahi kar sakta."

### Scene 4 — Pipeline overview (1:35–2:15)
- Visual: 6 boxes ka rail, orange dot left se right. Boxes 3, 5, 6 glow (neural nets).
  Document upar se aata hai, sawaal neeche se, dono box 3 pe milte hain.
- Screen text: "6 stages. 3 neural networks."
- Narration: "Andar chhe stage hain. Do stage plumbing hain: file se text, text se tukde.
  Teen stage neural networks hain: ek jo matlab samajhta hai, ek jo jawaab dhoondhta hai,
  aur ek optional jo use sentence mein likhta hai. Ek-ek karke dekhte hain."

### Scene 5 — Loader + Chunker (2:15–2:50)
- Visual: PDF, DOCX, XLSX, image icons ek funnel mein girte hain, bahar "blocks" nikalte
  hain: heading tag, paragraph tag, table-row tag, har ek pe "p.3" badge. Phir blocks
  ~180-word chunks mein pack hote hain, heading har chunk ke upar chipak jaati hai.
- Screen text: "Any file → blocks with page numbers → 180-word chunks that remember their section."
- Narration: "Pehle file ko text banate hain, lekin structure bachaate hain: heading kya
  hai, table ki row kya hai, kaunsa page. Phir 180 shabdon ke tukde banate hain kyunki
  reader model ek baar mein sirf 512 tokens padh sakta hai. Har tukda apna section aur page
  yaad rakhta hai. Isi se baad mein citation banti hai."

### Scene 6 — Embeddings + FAISS (2:50–3:40)
- Visual: chunk → neural box → 384 chhote numbers ka strip → ek dot 3D space mein.
  Kai dots. Sawaal "Which smart speakers does it work with?" → dot, jo "Alexa and Google
  Assistant" wale dot ke paas land karta hai. Side mein "TF-IDF" box: shabd "speaker"
  dhoondhta hai, kuch nahi milta, cross.
- Screen text: "Meaning → vector. Similar meaning → nearby. The word 'speaker' is not in the document."
- Narration: "Pehla neural network har tukde ko 384 numbers ke vector mein badalta hai.
  Jinka matlab same hai, unke vectors paas hote hain. Sawaal ko bhi vector banate hain aur
  FAISS sabse paas wale paanch tukde turant nikaal deta hai. Dekho: 'smart speakers' poocha,
  document mein 'speaker' shabd hai hi nahi, phir bhi Alexa wala tukda mila. Keyword search
  yeh kabhi nahi kar sakta. Humne purana keyword tareeka, TF-IDF, baseline ke taur pe rakha
  hai taaki fark naap sakein."

### Scene 7 — Reader: span extraction (3:40–4:40) ← sabse important scene
- Visual: token strip: `<s> How long is the warranty ? </s></s> The AquaBoil Pro is covered
  by a two - year limited warranty ...`. Har token ke upar do bars uthte hain: green
  (start logit), orange (end logit). "two" pe green sabse ooncha, "year" pe orange sabse
  ooncha. Dono ke beech ka hissa orange highlight. Formula fade in.
- Screen text: "score(i,j) = start[i] + end[j], j ≥ i. Best pair = answer."
- Narration: "Dusra neural network project ka dil hai. Sawaal aur ek tukda saath mein
  andar jaate hain. Transformer har token ke liye do numbers deta hai: start logit, matlab
  jawaab yahan se shuru hota hai, aur end logit, matlab yahan khatam. Hum har start-end
  pair ka score jodte hain, sabse bada score jawaab hai: 'two-year'. Yeh logic humne khud
  likha hai, reader.py mein, kisi library ke ready-made function se nahi."

### Scene 8 — "No answer" (4:40–5:15)
- Visual: wahi strip, sawaal "How much does it cost?". Ab pehle token `<s>` pe dono bars
  sabse oonche. Baaki sab chhote. `<s>` red outline. Screen pe "Not found in document".
- Screen text: "null score = start[0] + end[0]. If it wins, we abstain."
- Narration: "Aur yeh woh cheez hai jo chatbot nahi karta. Pehle token ka score 'jawaab
  nahi hai' ka score hai. Price document mein hai hi nahi, toh null score jeet jaata hai aur
  system bolta hai: not found. Humare test mein chhe ke chhe unanswerable sawaal sahi
  pakde gaye."

### Scene 9 — Generative toggle (5:15–5:45)
- Visual: UI toggle flip. Span "two-year" → teesra neural box → "The AquaBoil Pro is
  covered by a two-year limited warranty." Neeche chhota: "exact span: two-year".
- Screen text: "Optional: Flan-T5 rewrites the span into a sentence. Never without a span."
- Narration: "Teesra network optional hai. Reader ka span leke poora sentence likhta hai.
  Lekin rule sakht hai: span nahi mila toh yeh bhi chup rehta hai. Kyunki jab humne isse
  khud jawaab dene diya tha, isne kettle ka price 'one point two kilograms' bata diya tha."

### Scene 10 — Results (5:45–6:25)
- Visual: table animate: rows ek-ek karke. Phir chhota bar chart: top-k 3 vs 5, F1 0.78
  → 0.88. Phir model comparison: roberta-base 0.89 → deberta-v3-base 0.95.
- Screen text: "56 questions · 3 documents · EM 0.89 · F1 0.95 · abstain 6/6 · 0.28 s/question, CPU"
- Narration: "Humne 56 sawaal khud likhe, teen alag tarah ke documents pe, chhe aise jinka
  jawaab tha hi nahi. SQuAD wale metrics use kiye: Exact Match aur F1. Tuning guess se nahi,
  measurement se ki: top-k teen se paanch karne pe F1 point-seven-eight se point-eight-eight
  gaya. Phir chaar pretrained readers compare kiye; DeBERTa-v3-base ne point-nine-five diya,
  laptop CPU pe ek sawaal point-two-eight second mein."

### Scene 11 — "It's just a web app?" (6:25–7:00)
- Visual: UI screenshot shrink hoke ek chhoti window ban jaati hai; uske peeche bada
  pipeline diagram reveal hota hai, teen boxes glow.
- Screen text: "The page is 150 lines. The pipeline behind it is three neural networks."
- Narration: "Web page sirf input-output window hai. Project uske peeche hai: teen neural
  networks, structure-aware chunking, span extraction jo humne khud likha, abstain logic,
  aur ek evaluation framework jo har number reproduce karta hai. Sab laptop CPU pe, offline,
  bina kisi API ke."

### Scene 12 — Limitations + next (7:00–7:30)
- Visual: teen simple cards: "small self-written test set", "simple conversation memory",
  "OCR quality". Phir arrow: "Next: long real PDFs".
- Screen text: "Honest limits. Next: evaluate on 30-page real documents."
- Narration: "Limitations bhi saaf hain: test set chhota aur humara likha hua hai,
  conversation memory simple hai, scanned documents OCR pe depend karte hain. Agla kadam:
  lambe asli documents pe evaluation, jahan neural retriever ka fayda sabse zyada dikhega."

---

## Part C — Context jo code files mein nahi hai (decision log)

Yeh woh baatein hain jo humne kaam karte hue decide kiin ya seekhi. Video aur viva dono
mein yahi "humne kya achieve kiya" ka asli jawaab hai.

### C1. Yeh system kyun, chatbot kyun nahi
- Instructor ne sirf title diya tha: "Deep Learning-Based Document Question Answering".
- Humne decide kiya: system banayenge jisme UI patli layer ho aur DL pipeline asli project
  ho. Extractive QA chuna kyunki: hallucination nahi, evidence built-in, CPU pe chalta hai,
  aur SQuAD-style span prediction ek saaf supervised DL task hai jo samjhaya ja sakta hai.

### C2. Training nahi ki, aur kyun yeh sahi decision hai
- Reader models pehle hi SQuAD 2.0 (130k + 50k unanswerable) pe fine-tuned hain.
- Wahi base model wahi data pe dobara train karne se wahi result aata; sirf GPU time jalta.
- Training ka fayda tab hota jab apne domain ke hazaaron Q/A pairs hote. Nahi hain.
- Isliye humne training ki jagah **model selection by measurement** kiya: chaar readers apne
  test set pe compare kiye (neeche C7).
- Agar koi pooche "fine-tuning ka plan?": Track B plan mein likha hai, DistilBERT ko SQuAD
  subset pe Colab pe fine-tune karke before/after EM/F1 dikhana. Time ki wajah se stretch
  goal rakha.

### C3. Chunking se kya seekha
- Pehle simple sliding window tha (180 words, 40 overlap). Kaam karta tha.
- Phir structure-aware chunking banaya: headings chunk ke saath jaati hain, table row kabhi
  nahi kat-ti, page/section yaad rehta hai. Syllabus ke header sawaal (instructor, office,
  TA) jo pehle miss ho rahe the, sab sahi ho gaye. F1 0.86 → 0.97 us document pe.
- Lekin pehle version mein har section ka apna chhota chunk ban raha tha, aur chhote chunk
  pe reader overconfident ho jaata hai (kettle ka "colour" poocha toh "blue" bol diya, ring
  light wali line se). Fix: chhote sections ko ek chunk mein pack karo, heading inline rakho.
- Lesson: chunk boundary ek DL hyperparameter hai, data pe naapna padta hai.

### C4. Abstain threshold ka trade-off (sabse achha viva example)
- Reader ka null-margin knob banaya: positive margin = model kam abstain karta hai.
- Margin 3 pe F1 0.93 tak gaya, lekin system "How much does the kettle cost?" ka jawaab
  "1.7 liters" dene laga. Confident galat jawaab abstain se bahut bura hai.
- Decision: margin 0 rakha, F1 thoda kam lekin abstain 6/6.
- Alag se confidence floor 0.6 rakha: eval mein har galat jawaab 0.6 se neeche tha, har
  sahi jawaab 0.7 se upar. Yeh number data se aaya, hawa se nahi.

### C5. Generator ka hallucination aur grounded rule
- Pehle Flan-T5 ko retrieved passages dekar seedha jawaab likhne diya. Unanswerable sawaal
  pe usne "1.2 kilograms without water" bana diya (kettle ka weight, price ki jagah).
- Rule banaya: generator sirf tab chalega jab extractive reader ne span dhoondha ho; woh
  span sentence mein rewrite karega aur sentence mein span exactly hona chahiye, warna plain
  template fallback.
- Ek aur bug mila: prompt mein question context ke baad tha, lamba context tokenizer se
  truncate hota tha aur question kat jaata tha; model "Pretrained vision models" jaisa
  random jawaab deta tha. Fix: context ko shabdon mein kaat ke span ke aas-paas rakha.
- Suggested questions: model 12 banata hai, hum har ek ko reader se poochte hain, sirf
  wahi dikhate hain jinka confident jawaab hai. Isliye har chip click pe kaam karta hai.

### C6. Dense vs TF-IDF ka honest result
- Top-k 5 pe dono ka score same aaya. Wajah: test documents mein sirf 4–5 chunks hain,
  top-5 mein sab aa jaate hain, retriever ka koi role nahi bachta.
- Top-k 3 pe fark dikhta hai, aur wahan TF-IDF thoda aage tha (humare sawaal keyword-heavy
  the).
- Dense retriever ka asli fayda: paraphrase ("smart speakers") aur lambe documents (jahan
  50 chunks mein se 5 chunne hain). Isliye plan mein A4b hai: ek 20+ page asli PDF pe eval.
- Yeh "humne honestly report kiya" wala point hai; instructor ko yeh sunna achha lagta hai.

### C7. Model comparison (data/eval/models.md)
| Reader | Params | F1 | s/question |
|---|---|---|---|
| roberta-base-squad2 | 124M | 0.89 | 0.19 |
| deberta-v3-base-squad2 (default) | 184M | 0.95 | 0.28 |
| roberta-large-squad2 | 354M | 0.95 | 0.58 |
| deberta-v3-large-squad2 | 434M | 0.97 | 0.89 |
- Decision: deberta-v3-base default (sabse achha accuracy-per-cost, chhote host pe fit).
  Docker/local mein large via env variable.
- Yeh bhi bolo: DeBERTa-v3 RoBERTa se better kyun: disentangled attention (content aur
  position alag-alag attend hote hain) aur ELECTRA-style pretraining.

### C8. Known misses (chhupao mat, batao)
- "How do you enter pairing mode?" → reader Wi-Fi troubleshooting line deta hai, KEEP WARM
  wali instruction nahi. Demo mein yeh sawaal mat poocho; error analysis mein likho.
- Follow-up: "what does it do" after "what is SQuAD" → pehle galat topic pe gaya (koi
  memory nahi thi). Ab simple fix hai: chhota sawaal + pronoun ho toh pichla sawaal retrieval
  mein jod dete hain. "Who is the instructor?" → "where is their office?" ab kaam karta hai.
- Flan-T5-base chhota model hai: summary theek-thaak hai, kuch suggested questions vague.
  Span answers hi reliable core hain.

### C9. Engineering problems jo aaye
- transformers v5 ne ready-made question-answering pipeline hata diya. Humne span
  extraction khud likha (start/end logits, mask, argmax, null score). Project ke liye yeh
  achha hua: ab har line explain kar sakte hain.
- Streamlit Cloud pe torchvision error: file watcher transformers ke vision modules import
  karne ki koshish karta tha. Fix: watcher off.
- Streamlit config folder gitignore mein tha, isliye fixes deploy hi nahi ho rahe the. Fix:
  sirf secrets ignore karo.
- Streamlit free tier ~1 GB RAM: isliye reader base model, generator lazy-load, aur
  flan-t5-small ka fallback env variable.

### C10. Numbers (final, yaad rakhne wale)
- 56 questions, 3 documents (kettle manual, course syllabus, transformer overview), 6 unanswerable.
- EM 0.89, F1 0.95, Recall@5 1.00, abstain 6/6, 0.28 s/question CPU (deberta-v3-base).
- Pipeline: MiniLM (22M) + DeBERTa-v3-base (184M) + optional Flan-T5-base (250M).
- Sab offline, koi API nahi, koi GPU nahi.

---

## Part D — Har member ka 90-second script

**Member 1 (lead) — Problem, architecture, "not a web app"**
"Humara project Deep Learning-Based Document Question Answering hai. User koi bhi document
upload karta hai, sawaal poochta hai, system document ke andar se exact jawaab nikaalta hai,
evidence ke saath, aur jawaab na ho toh mana kar deta hai. Yeh chatbot nahi hai: extractive
QA hai. Andar chhe stages hain jisme teen neural networks hain: embedding model, reader, aur
optional generator. Web page sirf window hai; project uske peeche ka pipeline hai, jisme
span extraction aur abstain logic humne khud likha hai. Mere saathi har stage samjhayenge."

**Member 2 — Loader aur chunker**
"Pehla kaam file se text nikaalna hai, lekin structure ke saath: headings, paragraphs, table
rows, page numbers. PDF, Word, PowerPoint, Excel, scanned image via OCR, sab chalta hai.
Phir text ko 180 shabdon ke chunks mein baant-te hain kyunki transformer 512 tokens se zyada
nahi padh sakta. Har chunk apna section heading saath rakhta hai; isse syllabus ke header
sawaal, jo pehle miss ho rahe the, sab sahi hone lage. Chhote sections ko pack karna pada,
kyunki chhote chunk pe reader overconfident hota hai. Chunk boundary bhi ek hyperparameter hai."

**Member 3 — Retriever**
"Pehla neural network MiniLM hai, Sentence-BERT style. Har chunk aur sawaal 384 numbers ka
vector banta hai; same matlab, paas vector. FAISS cosine similarity se top-5 chunks nikaalta
hai. Isliye 'smart speakers' poochne pe Alexa wala chunk milta hai jabki 'speaker' shabd
document mein nahi hai. Humne TF-IDF keyword baseline bhi rakha hai. Chhote documents pe
dono same score dete hain kyunki top-5 mein saare chunks aa jaate hain; fark paraphrase aur
lambe documents pe dikhta hai. Yeh humne honestly report kiya."

**Member 4 — Reader**
"Dusra network DeBERTa-v3-base hai, SQuAD 2.0 pe fine-tuned. Sawaal aur chunk saath jaate
hain; model har token ke liye start logit aur end logit deta hai. Hum har start-end pair ka
score jodte hain, end start ke baad ho, span 40 tokens se chhota ho; sabse bada score jawaab.
Token zero ka score 'jawaab nahi hai' ka score hai; woh jeete toh abstain. Training loss
start aur end pe cross-entropy thi. Yeh poora logic humne reader.py mein khud likha hai
kyunki library ka ready-made function hat gaya tha."

**Member 5 — Evaluation aur generator**
"Humne 56 sawaal teen documents pe likhe, chhe unanswerable. Metrics: Exact Match, F1,
Recall@k, abstain accuracy, latency. Result: EM 0.89, F1 0.95, Recall 1.00, abstain 6/6,
0.28 second per question CPU pe. Tuning measurement se ki: top-k 3 se 5 pe F1 0.78 se 0.88;
abstain loose karne pe F1 0.93 lekin system price ka jawaab '1.7 liters' dene laga, isliye
strict rakha. Chaar readers compare kiye, DeBERTa-v3-base chuna. Generator Flan-T5 optional
hai aur sirf reader ke span ko sentence banata hai, kyunki free generation mein usne
hallucinate kiya tha."
