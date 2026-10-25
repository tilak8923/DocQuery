"""Fast tests that need no model downloads.

Run:  .venv/bin/python -m pytest tests/ -q   (or: python -m unittest discover tests)
"""

import unittest
from pathlib import Path

from dlqa.chunker import chunk_text
from dlqa.loader import load_document

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "transformers.txt"


class ChunkerTests(unittest.TestCase):
    def test_overlap_and_coverage(self):
        words = [f"w{i}" for i in range(500)]
        chunks = chunk_text(" ".join(words), chunk_words=100, overlap_words=20)
        self.assertEqual(chunks[0].text.split()[0], "w0")
        self.assertEqual(chunks[-1].text.split()[-1], "w499")
        # second chunk starts 80 words in (100 - 20 overlap)
        self.assertEqual(chunks[1].start_word, 80)
        # every chunk fits inside the limit
        self.assertTrue(all(len(c.text.split()) <= 100 for c in chunks))

    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("just a few words", chunk_words=100, overlap_words=20)
        self.assertEqual(len(chunks), 1)

    def test_bad_overlap_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_words=10, overlap_words=10)


class LoaderTests(unittest.TestCase):
    def test_txt_loads_and_cleans(self):
        text = load_document(SAMPLE)
        self.assertIn("Attention Is All You Need", text)
        self.assertNotIn("  ", text)

    def test_unsupported_type(self):
        with self.assertRaises(ValueError):
            load_document(Path(__file__))  # a .py file


if __name__ == "__main__":
    unittest.main()
