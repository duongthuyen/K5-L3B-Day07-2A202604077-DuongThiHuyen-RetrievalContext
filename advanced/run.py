"""Run the advanced benchmark for one config/split; save per-question traces + retrieval page metrics.

Configs (chosen on dev only):
- V0_text_dense   : text chunks only (no table/visual chunks), dense, no page viewing  -> plain-text RAG baseline
- V1_mm_hybrid    : text + table + visual-description chunks, hybrid RRF, no page viewing
- V2_mm_hybrid_view: V1 + same-page table/visual chunks + view_page tool (model checks original page image)
Retrieval metric (automatic, page level): for each gold page, did any context chunk come from it?
"All gold pages in context" = context evidence for single and multi-hop questions.
Pass/fail grading is manual (results/advanced/reviews/<run>.json).
Run: python -m advanced.run --split dev --config V2_mm_hybrid_view [--only AD01 ...] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from .agent import run
from .common import AGENT_MODEL, DATA, RESULTS
from .retrieval import build, load_chunks


class TextOnly:
    def __init__(self, inner):
        self.inner, self.name = inner, inner.name + "(text-only)"

    def search(self, query, *, filters=None, top_k=5):
        return self.inner.search(query, filters={"kind": "text"}, top_k=top_k)


CONFIGS = {
    "V0_text_dense":     {"method": "dense", "text_only": True, "view": False, "top_k": 5},
    "V1_mm_hybrid":      {"method": "hybrid", "text_only": False, "view": False, "top_k": 5},
    "V2_mm_hybrid_view": {"method": "hybrid+samepage", "text_only": False, "view": True, "top_k": 5},
    # Fixes from dev error analysis of V0-V2 (AD01 table split over 2 pages / table only as visual chunk,
    # AD06 prose final answer, AU02/AU04 'not found' without abstain flag). Note: the kind-group fix in
    # retrieval.py applies to V1/V2 re-runs as well.
    "V3_mm_hybrid_view_fix": {"method": "hybrid+samepage", "text_only": False, "view": True, "top_k": 5,
                              "adjacent_view": True, "json_repair": True},
    # V3 never opened a page (page_views_mean 0) although descriptions can be wrong (V1 AD12 misread month,
    # Huffman description misses leaf letters) -> force verification on the original page image.
    "V4_mm_hybrid_mustview": {"method": "hybrid+samepage", "text_only": False, "view": True, "top_k": 5,
                              "adjacent_view": True, "json_repair": True, "must_view": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=AGENT_MODEL)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}
    retriever = build(cfg["method"], chunks)
    if cfg["text_only"]:
        retriever = TextOnly(retriever)
    retriever.search("warm up", top_k=1)
    bench = [q for q in json.loads((DATA / "benchmark.json").read_text(encoding="utf-8")) if q["split"] == args.split]
    if args.only:
        bench = [q for q in bench if q["id"] in args.only]
    client = OpenAI()

    def one(q):
        try:
            r = run(q["question"], retriever, top_k=cfg["top_k"], allow_view=cfg["view"], model=args.model, client=client,
                    adjacent_view=cfg.get("adjacent_view", False), json_repair=cfg.get("json_repair", False),
                    must_view=cfg.get("must_view", False))
        except Exception as e:  # provider/store failure: counted as FAIL, reported separately
            r = {"status": "error", "error": f"{type(e).__name__}: {e}", "events": [], "citations": [], "context_ids": [],
                 "viewed_pages": [], "n_searches": 0, "usage": {}, "elapsed_s": 0, "cost_usd": 0}
        r = {"id": q["id"], "group": q["group"], "split": args.split, "config": args.config, **r}
        if q["group"] != "unanswerable":
            gold_pages = sorted({g["page"] for g in q["gold"]})
            ctx_pages = {by_id[i]["page"] for i in r["context_ids"]}
            r["gold_pages"] = gold_pages
            r["gold_pages_in_context"] = [p for p in gold_pages if p in ctx_pages]
            r["all_gold_pages_in_context"] = all(p in ctx_pages for p in gold_pages)
            first = next((e for e in r["events"] if e["event"] == "search"), None)
            r["first_search_gold_page_rank"] = next((x["rank"] for x in (first or {}).get("results", [])
                                                     if x["page"] in gold_pages and x["rank"]), None)
        return r

    with ThreadPoolExecutor(args.workers) as ex:
        traces = list(ex.map(one, bench))
    for t in traces:
        print(f"{t['id']:5} {t['group']:12} {t['status']:9} s={t['n_searches']} v={t.get('viewed_pages')} "
              f"ctx={t.get('all_gold_pages_in_context', '-')} {str(t.get('answer', t.get('error')))[:120]!r}")
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["group"] != "unanswerable"]
    summary = {"config": args.config, **cfg, "model": args.model, "split": args.split, "N": len(traces),
               "errors": len(traces) - len(ok), "status": {s: sum(t["status"] == s for t in traces) for s in ("answered", "abstain", "error")},
               "context_all_gold_pages": f"{sum(t['all_gold_pages_in_context'] for t in ans)}/{len(ans)}",
               "first_search_gold_page_at1": f"{sum(t['first_search_gold_page_rank'] == 1 for t in ans)}/{len(ans)}",
               "first_search_gold_page_at3": f"{sum((t['first_search_gold_page_rank'] or 99) <= 3 for t in ans)}/{len(ans)}",
               "searches_mean": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
               "page_views_mean": round(statistics.mean(len(t["viewed_pages"]) for t in ok), 2) if ok else None,
               "prompt_tokens": sum(t["usage"].get("prompt_tokens", 0) for t in ok),
               "completion_tokens": sum(t["usage"].get("completion_tokens", 0) for t in ok),
               "cost_usd": round(sum(t["cost_usd"] for t in ok), 4),
               "median_elapsed_s": statistics.median(t["elapsed_s"] for t in ok) if ok else None}
    out = RESULTS / f"{args.config}-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
