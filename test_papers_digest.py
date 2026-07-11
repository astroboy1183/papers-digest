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




class HfSignalTest(unittest.TestCase):

    def test_arxiv_id_extraction(self):
        self.assertEqual(pd.arxiv_id("http://arxiv.org/abs/2507.01234v2"), "2507.01234")
        self.assertEqual(pd.arxiv_id("https://arxiv.org/abs/2507.9876"), "2507.9876")
        self.assertEqual(pd.arxiv_id(""), "")
        self.assertEqual(pd.arxiv_id("https://example.com/paper"), "")

    def test_hot_flag_threshold(self):
        self.assertIn("🔥HF:50", pd.hot_flag({"upvotes": 50}))
        self.assertEqual(pd.hot_flag({"upvotes": 3}), "")
        self.assertEqual(pd.hot_flag({}), "")

    def test_hot_papers_force_included_after_skim(self):
        pool = papers(30)
        pool[25]["upvotes"] = 99  # trending but the skim will drop it
        saved = pd.ask_llm
        pd.ask_llm = lambda prompt, max_tokens=0, model="": "[0, 1]"
        try:
            kept = pd.shortlist(pool, "m")
        finally:
            pd.ask_llm = saved
        self.assertIn(pool[25], kept)
        self.assertEqual(len(kept), 3)


class ServedMemoryTest(unittest.TestCase):

    def test_split_state_extracts_picked_ids(self):
        text, ids = pd.split_state(
            'digest\n===STATE===\n{"picked": ["2507.01234", "2507.9"]}')
        self.assertEqual(text, "digest")
        self.assertEqual(ids, ["2507.01234", "2507.9"])

    def test_garbage_tail_costs_memory_not_digest(self):
        text, ids = pd.split_state("digest\n===STATE===\nnot json")
        self.assertEqual((text, ids), ("digest", []))
        text, ids = pd.split_state("no tail")
        self.assertEqual((text, ids), ("no tail", []))

    def test_served_prunes_and_survives_garbage(self):
        import tempfile
        from pathlib import Path as P
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as tmp:
            saved_dir, saved_file = pd.STATE_DIR, pd.SERVED_FILE
            pd.STATE_DIR = P(tmp)
            pd.SERVED_FILE = P(tmp) / "served.json"
            try:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                pd.save_served({"2507.1": today, "2401.9": "2024-01-01"})
                served = pd.load_served()
                self.assertIn("2507.1", served)
                self.assertNotIn("2401.9", served)
                pd.SERVED_FILE.write_text("junk")
                self.assertEqual(pd.load_served(), {})
            finally:
                pd.STATE_DIR, pd.SERVED_FILE = saved_dir, saved_file


class LinkGuardTest(unittest.TestCase):

    def test_known_link_kept_invented_stripped(self):
        known = {"http://arxiv.org/abs/2507.01234v1"}
        out = pd.validate_links(
            "A\nhttp://arxiv.org/abs/2507.01234v1\nB\nhttp://arxiv.org/abs/9999.11111v1",
            known)
        self.assertIn("2507.01234", out)
        self.assertNotIn("9999.11111", out)
        self.assertIn("(link unavailable)", out)

    def test_trailing_punctuation_tolerated(self):
        known = {"http://arxiv.org/abs/2507.01234v1"}
        out = pd.validate_links("See (http://arxiv.org/abs/2507.01234v1).", known)
        self.assertNotIn("(link unavailable)", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
