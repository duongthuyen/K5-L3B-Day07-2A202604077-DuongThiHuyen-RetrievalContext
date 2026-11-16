"""Offline unit tests for Day 7 code (src/). Fakes test control flow only, never quality."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.data import is_hit, norm
    from src.context import pack, expand_neighbours
    from src.retrievers import BM25Retriever, HybridRRF
    from src.agent import run_agent
    DEPS = True
except ModuleNotFoundError:  # running with a Python that lacks project deps (use .venv)
    DEPS = False

C = [{"id": "d:p0001:c00", "text": "An append() operation adds an element\nto the end of a list.", "source_id": "d",
      "source": "d.pdf#page=1", "page": 1, "metadata": {"book": "py", "chapter": 9, "section": "9.1"}},
     {"id": "d:p0001:c01", "text": "A pop() operation removes the last item.", "source_id": "d",
      "source": "d.pdf#page=1", "page": 1, "metadata": {"book": "py", "chapter": 9, "section": "9.1"}},
     {"id": "e:p0002:c00", "text": "Data cleaning removes duplicate values.", "source_id": "e",
      "source": "e.pdf#page=2", "page": 2, "metadata": {"book": "ds", "chapter": 2, "section": "2.4"}}]


class FakeDense:
    name = "fake_dense"
    def search(self, q, *, filters=None, top_k=3):
        return [{**c, "score": 1.0 / (i + 1), "rank": i + 1} for i, c in enumerate(reversed(C))][:top_k]


def fake_client(script):
    """script: list of ('tool', query) or ('final', json_str)."""
    it = iter(script)
    def create(**kw):
        kind, val = next(it)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        if kind == "tool":
            tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="search_corpus", arguments=json.dumps({"query": val})))
            msg = SimpleNamespace(content=None, tool_calls=[tc])
        else:
            msg = SimpleNamespace(content=val, tool_calls=None)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@unittest.skipUnless(DEPS, "project deps missing; run with .venv")
class Day7Tests(unittest.TestCase):
    def test_hit_normalizes_whitespace_and_checks_page(self):
        g = [{"source_id": "d", "page": 1, "evidence": "adds an element to the end"}]
        self.assertTrue(is_hit(C[0], g))
        self.assertFalse(is_hit(C[0], [{**g[0], "page": 2}]))
        self.assertEqual(norm(" a \n b "), "a b")

    def test_bm25_filter_and_rank(self):
        r = BM25Retriever(C)
        self.assertEqual(r.search("pop last item", top_k=1)[0]["id"], "d:p0001:c01")
        self.assertEqual(r.search("duplicate", filters={"book": "py"}), [])
        with self.assertRaises(ValueError):
            r.search("x", filters={"page": 1})

    def test_rrf_uses_ranks_not_raw_scores(self):
        h = HybridRRF(FakeDense(), BM25Retriever(C), candidates=3)
        out = h.search("pop last item", top_k=3)
        self.assertEqual(len({x["id"] for x in out}), len(out))
        self.assertIn("components", out[0])

    def test_pack_budget_and_dedup(self):
        ctx, ids = pack([C[0], C[0], C[1]], budget=10_000)
        self.assertEqual(ids, [C[0]["id"], C[1]["id"]])
        _, ids = pack(C, budget=120)
        self.assertLessEqual(len(ids), 1)
        self.assertEqual(len(expand_neighbours([C[0]], {c["id"]: c for c in C}, [c["id"] for c in C])), 2)

    def test_agent_answers_with_valid_citation(self):
        cid = "d:p0001:c00"
        client = fake_client([("tool", "append"), ("final", json.dumps({"answer": f"append() [{cid}]", "citations": [cid], "abstain": False}))])
        r = run_agent("q?", FakeDense(), client=client, model="gpt-4o-mini")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["n_searches"], 1)
        self.assertGreater(r["cost_usd"], 0)

    def test_agent_rejects_citation_not_in_context(self):
        client = fake_client([("tool", "x"), ("final", json.dumps({"answer": "y [zz:p0009:c00]", "citations": ["zz:p0009:c00"], "abstain": False}))])
        r = run_agent("q?", FakeDense(), client=client, top_k=1)
        self.assertEqual(r["status"], "abstain")
        self.assertIn("citations_not_in_context", r["abstain_reason"])

    def test_agent_canonicalizes_unpadded_citation(self):
        client = fake_client([("tool", "append"), ("final", json.dumps({"answer": "append() [d:p1:c0]", "citations": ["d:p1:c0"], "abstain": False}))])
        r = run_agent("q?", FakeDense(), client=client)
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"], ["d:p0001:c00"])

    def test_agent_search_limit_forces_final(self):
        client = fake_client([("tool", "a"), ("tool", "b"), ("final", json.dumps({"answer": "Không đủ bằng chứng để trả lời.", "citations": [], "abstain": True}))])
        r = run_agent("q?", FakeDense(), client=client, max_searches=2)
        self.assertEqual(r["n_searches"], 2)
        self.assertEqual(r["status"], "abstain")
        # second search returns only already-shown chunks -> dedup recorded
        self.assertTrue(r["events"][1]["duplicate_ids"])


if __name__ == "__main__":
    unittest.main()
