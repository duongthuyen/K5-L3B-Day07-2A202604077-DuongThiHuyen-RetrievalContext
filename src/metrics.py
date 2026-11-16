"""Aggregate agent traces + manual reviews for any split (wraps starter/agent_metrics.aggregate).
Run: python -m src.metrics --run A1_hybrid_neighbours-test2
"""
import argparse
import json
import sys

from .config import RESULTS_DIR, ROOT
from .data import load_cases, load_json, save_json

sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="<config>-<split>, e.g. A0_dense-dev")
    args = ap.parse_args()
    split = args.run.rsplit("-", 1)[1]
    traces = load_json(RESULTS_DIR / "agent" / args.run / "traces.json")
    reviews = load_json(RESULTS_DIR / "reviews" / f"{args.run}.json")
    report = {"run": args.run, "split": split, **aggregate(load_cases(split), traces, reviews)}
    save_json(RESULTS_DIR / "agent" / args.run / "agent_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
