"""Reader: a transformer that extracts the answer span from a passage.

Model: deepset/deberta-v3-base-squad2 by default (see data/eval/models.md for
the comparison; override with DLQA_READER_MODEL). Trained on SQuAD v2, where
some questions have no answer in the passage.

How span extraction works (this is the deep-learning core of the project):

1. The question and the passage are tokenized together:
       <s> question </s></s> passage </s>
2. RoBERTa produces a hidden vector for every token.
3. A linear layer on top turns each vector into two numbers: a START logit
   and an END logit. High START logit = "the answer begins here".
4. The best answer is the (start, end) pair with the highest
   start_logit + end_logit, with end >= start and both inside the passage.
5. The score of the pair (start=0, end=0), i.e. the <s> token, is the
   model's "no answer" score. If it beats the best real span, we return "".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import settings


@dataclass
class ReaderAnswer:
    text: str  # "" means the model thinks there is no answer in this passage
    score: float  # softmax probability of the chosen span (0..1)
    start_char: int
    end_char: int
    null_score: float = 0.0  # raw logit score of the "no answer" option
    span_score: float = 0.0  # raw logit score of the best real span


class ExtractiveReader:
    def __init__(
        self,
        model_name: str | None = None,
        max_answer_tokens: int = 40,
        null_margin: float = 0.0,
    ):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        model_name = model_name or settings.reader_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        self.max_answer_tokens = max_answer_tokens
        # Abstain only if null_score > span_score + null_margin. A positive
        # margin makes the reader less eager to say "no answer".
        self.null_margin = null_margin

    @torch.no_grad()
    def answer(self, question: str, context: str) -> ReaderAnswer:
        enc = self.tokenizer(
            question,
            context,
            truncation="only_second",  # cut the passage, never the question
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        out = self.model(**enc)
        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

        # sequence_ids: None for special tokens, 0 for question, 1 for context
        seq_ids = enc.sequence_ids(0)
        ctx_idx = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx_idx:
            return ReaderAnswer("", 0.0, 0, 0)
        c0, c1 = ctx_idx[0], ctx_idx[-1]

        # Best span restricted to the context part, length-limited.
        s_logits = start_logits[c0 : c1 + 1]
        e_logits = end_logits[c0 : c1 + 1]
        n = s_logits.shape[0]
        scores = s_logits[:, None] + e_logits[None, :]
        mask = torch.ones(n, n, dtype=torch.bool).triu()  # end >= start
        mask &= ~torch.ones(n, n, dtype=torch.bool).triu(self.max_answer_tokens)
        scores = scores.masked_fill(~mask, float("-inf"))
        best = int(torch.argmax(scores))
        bs, be = divmod(best, n)
        span_score = float(scores[bs, be])

        null_score = float(start_logits[0] + end_logits[0])

        # Probability-style confidence: softmax over {null, best span}.
        prob = float(torch.softmax(torch.tensor([span_score, null_score]), dim=0)[0])

        if null_score > span_score + self.null_margin:
            return ReaderAnswer("", prob, 0, 0, null_score=null_score, span_score=span_score)

        start_char = offsets[c0 + bs][0]
        end_char = offsets[c0 + be][1]
        text = context[start_char:end_char].strip()
        return ReaderAnswer(
            text=text,
            score=prob,
            start_char=start_char,
            end_char=end_char,
            null_score=null_score,
            span_score=span_score,
        )
