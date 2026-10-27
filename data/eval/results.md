# Evaluation results (top-k = 5, null-margin = 0.0, chunk = 180 words)

| Retriever | Document | N | EM | F1 | Recall@k | Abstain acc | Latency (s) |
|---|---|---|---|---|---|---|---|
| dense | kettle_manual.txt | 19 | 0.89 | 0.92 | 1.00 | 1.00 | 0.23 |
| dense | syllabus.txt | 18 | 0.83 | 0.86 | 1.00 | 1.00 | 0.16 |
| dense | transformers.txt | 19 | 0.89 | 0.89 | 1.00 | 1.00 | 0.18 |
| **dense** | **overall** | 56 | **0.88** | **0.89** | **1.00** | **1.00** | **0.19** |
| tfidf | kettle_manual.txt | 19 | 0.89 | 0.92 | 1.00 | 1.00 | 0.19 |
| tfidf | syllabus.txt | 18 | 0.83 | 0.86 | 1.00 | 1.00 | 0.15 |
| tfidf | transformers.txt | 19 | 0.89 | 0.89 | 1.00 | 1.00 | 0.16 |
| **tfidf** | **overall** | 56 | **0.88** | **0.89** | **1.00** | **1.00** | **0.17** |

EM/F1 use SQuAD normalisation. Unanswerable questions count as EM=1 when the system abstains.
Recall@k is computed on answerable questions only.

## Misses (F1 < 0.5)

- `dense` / kettle_manual.txt: **How often should the kettle be descaled?** → `` (gold: ['every four to eight weeks', 'four to eight weeks'])
- `dense` / syllabus.txt: **What is the late penalty?** → `` (gold: ['10 percent per day'])
- `dense` / syllabus.txt: **Which Python version is used?** → `CS 3310` (gold: ['Python 3.10', '3.10'])
- `dense` / transformers.txt: **What did the authors add to give the model a notion of word order?** → `` (gold: ['positional encodings'])
- `dense` / transformers.txt: **What percentage of input tokens are masked in BERT pretraining?** → `` (gold: ['15 percent'])
- `tfidf` / kettle_manual.txt: **How often should the kettle be descaled?** → `` (gold: ['every four to eight weeks', 'four to eight weeks'])
- `tfidf` / syllabus.txt: **What is the late penalty?** → `` (gold: ['10 percent per day'])
- `tfidf` / syllabus.txt: **Which Python version is used?** → `CS 3310` (gold: ['Python 3.10', '3.10'])
- `tfidf` / transformers.txt: **What did the authors add to give the model a notion of word order?** → `` (gold: ['positional encodings'])
- `tfidf` / transformers.txt: **What percentage of input tokens are masked in BERT pretraining?** → `` (gold: ['15 percent'])
