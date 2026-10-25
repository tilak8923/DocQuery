#!/bin/bash
# Quick end-to-end smoke run over the sample document.
Q=(
"Who proposed the Transformer?"
"How many parameters does BERT-base have?"
"What are the two pretraining objectives of BERT?"
"How long did training take on WMT 2014?"
"Who developed RoBERTa?"
"What does SQuAD 2.0 add?"
"What is the capital of France?"
)
for q in "${Q[@]}"; do
  echo "Q: $q"
  .venv/bin/python -m dlqa.cli data/samples/transformers.txt "$q" 2>/dev/null | grep -E "^(Answer|Confidence|Time)"
  echo
done
