# Evaluation results (top-k = 5, null-margin = 0.0, chunk = 180 words)

| Retriever | Document | N | EM | F1 | Recall@k | Abstain acc | Latency (s) |
|---|---|---|---|---|---|---|---|
| dense | kettle_manual.txt | 19 | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 |
| dense | syllabus.txt | 18 | 0.83 | 0.91 | 1.00 | 1.00 | 0.25 |
| dense | transformers.txt | 19 | 0.84 | 0.95 | 1.00 | 1.00 | 0.29 |
| **dense** | **overall** | 56 | **0.89** | **0.95** | **1.00** | **1.00** | **0.31** |

EM/F1 use SQuAD normalisation. Unanswerable questions count as EM=1 when the system abstains.
Recall@k is computed on answerable questions only.

## Misses (F1 < 0.5)

- `dense` / syllabus.txt: **How much is each programming assignment worth?** → `40 percent of the final grade` (gold: ['10 percent'])
