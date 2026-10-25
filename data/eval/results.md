# Evaluation results (top-k = 5, null-margin = 0.0, chunk = 180 words)

| Retriever | Document | N | EM | F1 | Recall@k | Abstain acc | Latency (s) |
|---|---|---|---|---|---|---|---|
| dense | kettle_manual.txt | 19 | 0.84 | 0.87 | 1.00 | 1.00 | 0.23 |
| dense | syllabus.txt | 18 | 0.83 | 0.86 | 1.00 | 1.00 | 0.20 |
| dense | transformers.txt | 19 | 0.89 | 0.89 | 1.00 | 1.00 | 0.19 |
| **dense** | **overall** | 56 | **0.86** | **0.88** | **1.00** | **1.00** | **0.21** |
| tfidf | kettle_manual.txt | 19 | 0.84 | 0.87 | 1.00 | 1.00 | 0.23 |
| tfidf | syllabus.txt | 18 | 0.83 | 0.86 | 1.00 | 1.00 | 0.19 |
| tfidf | transformers.txt | 19 | 0.89 | 0.89 | 1.00 | 1.00 | 0.19 |
| **tfidf** | **overall** | 56 | **0.86** | **0.88** | **1.00** | **1.00** | **0.20** |

EM/F1 use SQuAD normalisation. Unanswerable questions count as EM=1 when the system abstains.
Recall@k is computed on answerable questions only.

## Misses (F1 < 0.5)

- `dense` / kettle_manual.txt: **How long does keep warm last?** → `` (gold: ['up to 30 minutes', '30 minutes'])
- `dense` / kettle_manual.txt: **What should be mixed with water for descaling?** → `` (gold: ['white vinegar', 'equal parts water and white vinegar'])
- `dense` / syllabus.txt: **What is the late penalty?** → `` (gold: ['10 percent per day'])
- `dense` / syllabus.txt: **How many students are in a project team?** → `` (gold: ['two or three'])
- `dense` / transformers.txt: **What did the authors add to give the model a notion of word order?** → `` (gold: ['positional encodings'])
- `dense` / transformers.txt: **What percentage of input tokens are masked in BERT pretraining?** → `` (gold: ['15 percent'])
- `tfidf` / kettle_manual.txt: **How long does keep warm last?** → `` (gold: ['up to 30 minutes', '30 minutes'])
- `tfidf` / kettle_manual.txt: **What should be mixed with water for descaling?** → `` (gold: ['white vinegar', 'equal parts water and white vinegar'])
- `tfidf` / syllabus.txt: **What is the late penalty?** → `` (gold: ['10 percent per day'])
- `tfidf` / syllabus.txt: **How many students are in a project team?** → `` (gold: ['two or three'])
- `tfidf` / transformers.txt: **What did the authors add to give the model a notion of word order?** → `` (gold: ['positional encodings'])
- `tfidf` / transformers.txt: **What percentage of input tokens are masked in BERT pretraining?** → `` (gold: ['15 percent'])
