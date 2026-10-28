"""Generator: a small sequence-to-sequence model (Flan-T5) for the
generative mode, document summaries and suggested questions.

It is only loaded when first used, so the extractive path stays light.
Answers are grounded: the model only sees the retrieved passages, and the
prompt tells it to say when the answer is not there.
"""

from __future__ import annotations

import re

import torch

from .config import settings

_NOT_FOUND = "not found in document"


def _window(contexts: list[str], span: str, max_words: int) -> str:
    """Join contexts and cut them to max_words, keeping the span in the middle."""
    text = " ".join(" ".join(c.split()) for c in contexts if c.strip())
    words = text.split()
    if len(words) <= max_words:
        return text
    if span and span in text:
        char_pos = text.index(span)
        start_word = len(text[:char_pos].split())
        lo = max(0, start_word - max_words // 2)
        return " ".join(words[lo : lo + max_words])
    return " ".join(words[:max_words])


class Generator:
    def __init__(self, model_name: str | None = None, max_input_tokens: int = 768):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = model_name or settings.gen_model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = self._load(AutoModelForSeq2SeqLM, name)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.name = name

    @staticmethod
    def _load(cls, name: str):
        """Load the seq2seq model and make sure every weight is real.

        transformers >= 5 builds the model on the "meta" device and then fills
        the weights in. For T5-family checkpoints whose encoder/decoder token
        embeddings are tied to ``shared`` this has left ``embed_tokens`` on the
        meta device on some hosts (seen on Streamlit Community Cloud), and the
        first ``generate()`` then fails with
        "Tensor.item() cannot be called on meta tensors". Re-tie the
        embeddings, and if anything is still on meta, reload from the
        ``pytorch_model.bin`` checkpoint, which stores every key explicitly.
        """
        model = cls.from_pretrained(name, dtype=torch.float32)
        try:
            model.tie_weights()
            shared = model.get_input_embeddings()
            for part in ("encoder", "decoder"):
                sub = getattr(model, part, None)
                if sub is not None and hasattr(sub, "embed_tokens"):
                    sub.embed_tokens = shared
        except Exception:  # noqa: BLE001 - best effort, checked below
            pass
        meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        meta += [n for n, b in model.named_buffers() if b.device.type == "meta"]
        if meta:
            model = cls.from_pretrained(name, dtype=torch.float32, use_safetensors=False)
            model.tie_weights()
            still = [n for n, p in model.named_parameters() if p.device.type == "meta"]
            if still:
                raise RuntimeError(
                    f"{name}: weights not loaded (still on meta device): {still[:5]}"
                )
        return model

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 96, **kw) -> list[str]:
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        )
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=kw.pop("num_beams", 3),
            no_repeat_ngram_size=3,
            early_stopping=True,
            **kw,
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]

    # ------------------------------------------------------------ answers
    def answer(self, question: str, contexts: list[str], span_hint: str = "") -> str:
        """Write a full-sentence answer from the retrieved passages.

        span_hint is the extractive reader's span (if any); giving it to the
        generator keeps the generated sentence anchored on the same fact.
        """
        # Keep the context short enough that the prompt never gets truncated
        # (which would cut off the question). Centre it on the span if we have one.
        context = _window(contexts, span_hint, max_words=220)
        if span_hint:
            # Sentence-rewrite task: small seq2seq models do this reliably, and the
            # exact fact is guaranteed to be present.
            prompt = (
                "Write one complete sentence that answers the question and contains the fact "
                "exactly as given.\n"
                f"Question: {question}\nFact: {span_hint}\nContext: {context}\nSentence:"
            )
            text = self._generate(prompt, max_new_tokens=64)[0]
            # Guard: if the model dropped the fact, fall back to a plain template.
            if span_hint.lower() not in text.lower():
                text = f"{span_hint}."
            return text
        prompt = (
            f"Question: {question}\n"
            "Using only the context below, answer the question in one complete sentence. "
            f"If the context does not contain the answer, reply exactly: {_NOT_FOUND}.\n\n"
            f"Context: {context}\n\nAnswer:"
        )
        text = self._generate(prompt, max_new_tokens=64)[0]
        return "" if _NOT_FOUND in text.lower() or not text else text

    # ------------------------------------------------------------ overview
    def summarize(self, text: str, max_words: int = 600) -> str:
        head = " ".join(text.split()[:max_words])
        prompt = f"Summarize the following document in three sentences.\n\n{head}"
        return self._generate(prompt, max_new_tokens=120)[0]

    def suggest_questions(self, text: str, n: int = 4, max_words: int = 500) -> list[str]:
        head = " ".join(text.split()[:max_words])
        prompt = (
            "Write one short factual question (starting with who, what, when, where, "
            "how many or how long) whose answer is stated in the text below.\n\n"
            f"Text: {head}\n\nQuestion:"
        )
        outs = self._generate(
            prompt,
            max_new_tokens=32,
            num_beams=1,
            do_sample=True,
            top_p=0.92,
            temperature=0.9,
            num_return_sequences=n * 2,
        )
        seen, qs = set(), []
        for q in outs:
            q = q.strip().strip('"')
            if not q.endswith("?"):
                q += "?"
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if len(q.split()) >= 3 and key not in seen:
                seen.add(key)
                qs.append(q[0].upper() + q[1:])
            if len(qs) == n:
                break
        return qs
