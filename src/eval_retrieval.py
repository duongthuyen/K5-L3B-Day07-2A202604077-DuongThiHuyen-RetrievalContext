"""Retrieval + context benchmark (no LLM). Same corpus/index/questions/labels for every config.

Hit@1, Evidence Hit@3: valid evidence in top-1 / top-3 (always computed on top-3 ranking).
Context Evidence Rate: valid evidence inside the final packed context / answerable questions.
Run: python -m src.eval_retrieval --split dev [--configs R0_dense R1_bm25 ...]
"""
from __future__ import annotations

import argparse
import statistics
import time

from .config import RESULTS_DIR
from .context import expand_neighbours, pack
from .data import is_hit, load_chunks, load_questions, save_json
from .retrievers import build

# Frozen config table. top_k = chunks put in context; ranking metrics always use top-3.
CONFIGS = {
    "R0_dense":         {"method": "dense",         "top_k": 3, "budget": 3000, "neighbours": False, "use_filter": True,
                         "note": "Baseline = Day 6 chosen config C (B chunks + user filter) as a retriever"},
    "R1_bm25":          {"method": "bm25",          "top_k": 3, "budget": 3000, "neighbours": False, "use_filter": True,
                         "note": "Lexical: exact function names (sum(), pop()) should match"},
    "R2_hybrid":        {"method": "hybrid",        "top_k": 3, "budget": 3000, "neighbours": False, "use_filter": True,
                         "note": "Improvement 1: RRF(dense, bm25) recovers questions where one retriever misses"},
    "R3_hybrid_rerank": {"method": "hybrid_rerank", "top_k": 3, "budget": 3000, "neighbours": False, "use_filter": True,
                         "note": "Improvement 2: cross-encoder rerank of 20 RRF candidates fixes ordering (Hit@1)"},
    "R4_dense_rerank":  {"method": "dense_rerank",  "top_k": 3, "budget": 3000, "neighbours": False, "use_filter": True,
                         "note": "Ablation: rerank without lexical candidates"},
    "R5_hybrid_rerank_k5": {"method": "hybrid_rerank", "top_k": 5, "budget": 4500, "neighbours": False, "use_filter": True,
                         "note": "Improvement 3: more context (top-5, 4500 chars) for evidence at rank 4-5"},
    "R6_dense_nofilter": {"method": "dense",        "top_k": 3, "budget": 3000, "neighbours": False, "use_filter": False,
                         "note": "Ablation: Day 6 config B (no filter)"},
    # Added after dev error analysis of R0-R5 (q09, q13, t02: right page, adjacent chunk).
    "R7_dense_neighbours":  {"method": "dense",  "top_k": 3, "budget": 6000, "neighbours": True, "use_filter": True,
                         "note": "Improvement 4: dense + prev/next chunk of same page, budget 6000"},
    "R8_hybrid_neighbours": {"method": "hybrid", "top_k": 3, "budget": 6000, "neighbours": True, "use_filter": True,
                         "note": "Improvement 5: hybrid RRF + prev/next chunk of same page, budget 6000"},
}


def run(config_name: str, split: str, chunks, questions) -> dict:
    cfg = CONFIGS[config_name]
    retriever = build(cfg["method"], chunks)
    by_id = {c["id"]: c for c in chunks}
    ordered = [c["id"] for c in chunks]
    retriever.search("warm up", top_k=1)  # load models before timing
    traces = []
    for q in questions:
        filters = q.get("filter") if cfg["use_filter"] else None
        t0 = time.perf_counter()
        err = None
        try:
            results = retriever.search(q["question"], filters=filters, top_k=max(3, cfg["top_k"]))
        except Exception as e:  # report, never fake
            results, err = [], f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        top3 = results[:3]
        ctx_src = results[:cfg["top_k"]]
        if cfg["neighbours"]:
            ctx_src = expand_neighbours(ctx_src, by_id, ordered)
        context, ctx_ids = pack(ctx_src, cfg["budget"])
        marked = [{"rank": r["rank"], "id": r["id"], "source": r["source"], "page": r["page"],
                   "score": r["score"], "hit": is_hit(r, q["gold"]),
                   **({"components": r["components"]} if "components" in r else {}),
                   **({"first_stage_rank": r["first_stage_rank"]} if "first_stage_rank" in r else {})}
                  for r in results[:cfg["top_k"]]]
        traces.append({
            "query_id": q["id"], "split": split, "type": q.get("type"), "question": q["question"],
            "query_sent": q["question"], "filter": filters, "config": config_name, "error": err,
            "results": marked, "context_ids": ctx_ids, "context_chars": len(context),
            "hit1": bool(top3 and is_hit(top3[0], q["gold"])),
            "hit3": any(is_hit(r, q["gold"]) for r in top3),
            "context_hit": any(is_hit(by_id[i], q["gold"]) for i in ctx_ids),
            "latency_ms": round(ms, 1),
        })
    n = len(traces)
    summary = {"config": config_name, **cfg, "split": split, "N": n,
               "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
               "context_hit": sum(t["context_hit"] for t in traces),
               "errors": sum(bool(t["error"]) for t in traces),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
               "cost_usd": 0.0}
    for k in ("hit1", "hit3", "context_hit"):
        summary[k + "_rate"] = round(summary[k] / n, 3)
    out = RESULTS_DIR / "retrieval" / f"{config_name}-{split}"
    save_json(out / "traces.json", traces)
    save_json(out / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test", "test2"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    chunks = load_chunks()
    questions = load_questions(args.split)
    rows = [run(c, args.split, chunks, questions) for c in args.configs]
    print(f"{'config':22} {'Hit@1':>9} {'Hit@3':>9} {'CtxEv':>9} {'ctx_ch':>7} {'ms':>7}")
    for s in rows:
        print(f"{s['config']:22} {s['hit1']:>3}/{s['N']:<2}{s['hit1_rate']:.2f} {s['hit3']:>3}/{s['N']:<2}{s['hit3_rate']:.2f}"
              f" {s['context_hit']:>3}/{s['N']:<2}{s['context_hit_rate']:.2f} {s['avg_context_chars']:>7} {s['median_latency_ms']:>7}")
    save_json(RESULTS_DIR / "retrieval" / f"comparison-{args.split}.json", rows)


if __name__ == "__main__":
    main()
