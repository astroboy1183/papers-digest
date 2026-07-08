#!/usr/bin/env python3
"""Offline unit tests for papers_digest — no network, no model.

Covers the stage-1 shortlist mechanics: index parsing, recall-biased
fallback on unparseable replies, out-of-range index filtering, and
chunking.
"""

import unittest

import papers_digest as pd


def papers(n):
    return [
        {"category": "cs.LG", "title": f"Paper {i}", "abstract": "A" * 300,
         "link": f"https://arxiv.org/abs/{i}"}
        for i in range(n)
    ]


class ShortlistTest(unittest.TestCase):

    def _run(self, pool, replies, chunk=None):
        calls = []

        def fake_ask(prompt, max_tokens=0, model=""):
            calls.append(prompt)
            return replies[len(calls) - 1]

        saved_ask, saved_chunk = pd.ask_llm, pd.FILTER_CHUNK
        pd.ask_llm = fake_ask
        if chunk:
            pd.FILTER_CHUNK = chunk
        try:
            kept = pd.shortlist(pool, "model-x")
        finally:
            pd.ask_llm, pd.FILTER_CHUNK = saved_ask, saved_chunk
        return kept, len(calls)

    def test_keeps_only_returned_indices(self):
        kept, _ = self._run(papers(5), ["[0, 3]"])
        self.assertEqual([p["title"] for p in kept], ["Paper 0", "Paper 3"])

    def test_unparseable_reply_keeps_whole_chunk(self):
        kept, _ = self._run(papers(4), ["sorry, here are my thoughts"])
        self.assertEqual(len(kept), 4)  # recall over precision

    def test_out_of_range_indices_filtered(self):
        kept, _ = self._run(papers(3), ["[0, 7, -1, 2]"])
        self.assertEqual([p["title"] for p in kept], ["Paper 0", "Paper 2"])

    def test_chunking_makes_one_call_per_chunk(self):
        kept, calls = self._run(papers(5), ["[0]", "[1]", "[0]"], chunk=2)
        self.assertEqual(calls, 3)  # 2 + 2 + 1
        self.assertEqual(len(kept), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
