"""Run the real agent on one split for one agent config; saves full traces.

Run: python -m src.run_agent --split dev --config A0_dense
Outputs results/agent/<config>-<split>/traces.json (+ summary.json with automatic,
non-semantic metrics: tool use, #searches, context evidence rate, time, cost).
Answer correctness / citation support / abstention are labelled by hand in reviews.json.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from openai import OpenAI

from .agent import run_agent
from .config import AGENT_MODEL, RESULTS_DIR
from .data import is_hit, load_cases, load_chunks, load_json, save_json
from .config import QUESTIONS_PATH
from .retrievers import build

AGENT_CONFIGS = {
    "A0_dense":         {"method": "dense",         "top_k": 3, "max_searches": 3, "budget": 8000,
                         "note": "Baseline agent: dense pgvector tool (Day 6 config C)"},
    "A1_hybrid_neighbours": {"method": "hybrid+neighbours", "top_k": 3, "max_searches": 3, "budget": 8000,
                         "note": "Selected on dev (R8): hybrid RRF tool + same-page neighbour chunks"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", required=True, choices=list(AGENT_CONFIGS))
    ap.add_argument("--only", nargs="*", help="subset of case ids (debug)")
    args = ap.parse_args()
    cfg = AGENT_CONFIGS[args.config]
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}
    gold = {q["id"]: q["gold"] for q in load_json(QUESTIONS_PATH)}  # used ONLY for scoring after the run
    retriever = build(cfg["method"], chunks)
    retriever.search("warm up", top_k=1)
    client = OpenAI()
    traces = []
    for case in load_cases(args.split):
        if args.only and case["id"] not in args.only:
            continue
        try:
            r = run_agent(case["question"], retriever, filters=case.get("filter"), top_k=cfg["top_k"],
                          max_searches=cfg["max_searches"], budget=cfg["budget"], client=client)
        except Exception as e:  # provider / vector store failure: record, never replace with fake output
            r = {"status": "error", "error": f"{type(e).__name__}: {e}", "events": [], "citations": [],
                 "context_ids": [], "n_searches": 0, "usage": {}, "elapsed_s": 0, "cost_usd": 0}
        r = {"query_id": case["id"], "split": args.split, "answerable": case["answerable"], "config": args.config, **r}
        if case["answerable"]:
            r["context_evidence"] = any(is_hit(by_id[i], gold[case["id"]]) for i in r["context_ids"])
            r["first_search_hit3"] = next((any(is_hit(by_id[x["id"]], gold[case["id"]]) for x in e["results"][:3])
                                           for e in r["events"] if e["event"] == "search"), False)
        print(f"{case['id']:4} {r['status']:9} searches={r['n_searches']} ctx_ev={r.get('context_evidence', '-')}"
              f" {r.get('answer', r.get('error', ''))[:110]!r}")
        traces.append(r)
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    summary = {
        "config": args.config, **cfg, "model": AGENT_MODEL, "split": args.split, "cases": len(traces),
        "errors": len(traces) - len(ok),
        "tool_use": {"n": sum(t["n_searches"] > 0 for t in ok), "N": len(ok)},
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "context_evidence_rate": {"n": sum(t["context_evidence"] for t in ans), "N": len(ans)},
        "status": dict(Counter(t["status"] for t in traces)),
        "llm_calls": sum(t["usage"].get("llm_calls", 0) for t in ok),
        "prompt_tokens": sum(t["usage"].get("prompt_tokens", 0) for t in ok),
        "completion_tokens": sum(t["usage"].get("completion_tokens", 0) for t in ok),
        "cost_usd": round(sum(t["cost_usd"] for t in ok), 5),
        "median_elapsed_s": statistics.median(t["elapsed_s"] for t in ok) if ok else None,
        "total_elapsed_s": round(sum(t["elapsed_s"] for t in ok), 1),
    }
    out = RESULTS_DIR / "agent" / f"{args.config}-{args.split}"
    save_json(out / "traces.json", traces)
    save_json(out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
