"""Retrieval-only comparison on the advanced benchmark (original question as query, no LLM).

Hit = a retrieved chunk lies on a gold page. For multi-hop questions, Hit@k counts only when
ALL gold pages are covered within the top-k ("all pages" is stricter than any page).
Run: python -m advanced.eval_retrieval --split dev
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from .common import DATA, RESULTS
from .retrieval import build, load_chunks
from .run import TextOnly

METHODS = {"text_dense": ("dense", True), "text_bm25": ("bm25", True), "mm_dense": ("dense", False),
           "mm_bm25": ("bm25", False), "mm_hybrid": ("hybrid", False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    chunks = load_chunks()
    bench = [q for q in json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
             if q["split"] == args.split and q["group"] != "unanswerable"]
    rows, traces = [], {}
    for name, (method, text_only) in METHODS.items():
        r = build(method, chunks)
        r = TextOnly(r) if text_only else r
        r.search("warm up", top_k=1)
        h1 = h3 = hk = 0
        lat, tr = [], []
        for q in bench:
            gold = {g["page"] for g in q["gold"]}
            t0 = time.perf_counter()
            res = r.search(q["question"], top_k=args.k)
            lat.append((time.perf_counter() - t0) * 1000)
            pages = [x["page"] for x in res]
            cov = lambda k: gold <= set(pages[:k])  # noqa: E731
            h1 += cov(1); h3 += cov(3); hk += cov(args.k)
            tr.append({"id": q["id"], "group": q["group"], "gold_pages": sorted(gold),
                       "top": [(x["id"], x["kind"], x.get("label")) for x in res], "hit1": cov(1), "hit3": cov(3), f"hit{args.k}": cov(args.k)})
        n = len(bench)
        rows.append({"method": name, "N": n, "hit1": f"{h1}/{n}", "hit3": f"{h3}/{n}", f"hit{args.k}": f"{hk}/{n}",
                     "median_ms": round(statistics.median(lat), 1)})
        traces[name] = tr
        print(rows[-1])
    out = RESULTS / f"retrieval-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
