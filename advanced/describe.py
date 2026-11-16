"""Step 2 — describe visual content with a vision model (gpt-4.1-mini), page by page.

Only pages that have a table, a raster image, a vector-drawing region or a Figure/Table caption
are sent. The model receives the page render + the page's own text layer (so labels it reads can be
cross-checked) and returns JSON: one entry per non-decorative figure/chart/diagram/table with
label, kind, title, axes/units/legend, the values it can read, and a faithful description.
Descriptions are DATA for retrieval; the agent can always re-check the original page image
with its view_page tool, and benchmark grading is done against the original PDF.
Cached per page in advanced/data/visual/pNNN.json (re-runs skip done pages).
Run: python -m advanced.describe [--workers 8]
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .common import CACHE, DATA, VISION_MODEL, cost

PROMPT = """You index a page of the textbook "Principles of Data Science" (OpenStax) for retrieval.
Look at the page image. For EVERY informative visual element on it (chart/graph/plot, diagram/flowchart/tree,
screenshot or photo that carries information, table), output one JSON object. Skip purely decorative
chapter-opening photos (say so with kind "decorative").
Fields:
- "label": e.g. "Figure 5.8" or "Table 2.3" (from the caption; null if none)
- "kind": "chart" | "diagram" | "image" | "table" | "decorative"
- "title": caption / chart title
- "details": for charts: chart type, x-axis label & range/categories, y-axis label, units & range, legend/series,
  and EVERY value you can read (bar labels, cell values, peaks, minimum/maximum and where they occur, trend).
  For diagrams: every node/box label and how they are connected (arrows, edge labels, order, branches).
  For tables: the full table as markdown with header, units and every row.
  For images/screenshots: what is shown and any readable text.
- "summary": 1-2 sentences of what the element shows / why it is there.
Be faithful: only state what is visible; write "unreadable" instead of guessing.
Return ONLY {"elements": [ ... ]}.
Page text layer (for cross-checking labels):
"""


def describe_page(client: OpenAI, page: dict) -> dict:
    img = base64.b64encode((CACHE / page["render"]).read_bytes()).decode()
    t0 = time.perf_counter()
    last_err = None
    for attempt in range(3):  # dense charts can make the model list hundreds of points -> cap + retry
        try:
            resp = client.with_options(timeout=180).chat.completions.create(
                model=VISION_MODEL, temperature=0, response_format={"type": "json_object"}, max_tokens=3500,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT + ("\nKeep 'details' under 400 words; summarize long data series by "
                                                       "range, peaks and trend." if attempt else "") + page["text"][:6000]},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": "high"}}]}])
            elements = json.loads(resp.choices[0].message.content).get("elements", [])
            elements = [e if isinstance(e, dict) else {"kind": "image", "label": None, "details": str(e)} for e in elements]
            break
        except Exception as e:  # timeout / truncated JSON -> retry with a shorter-output instruction
            last_err = e
    else:
        raise RuntimeError(f"failed after 3 attempts: {last_err}")
    u = resp.usage
    return {"page": page["page"], "model": VISION_MODEL, "elements": elements, "attempts": attempt + 1,
            "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens},
            "cost_usd": round(cost(VISION_MODEL, u.prompt_tokens, u.completion_tokens), 6),
            "elapsed_s": round(time.perf_counter() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = DATA / "visual"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = [json.loads(l) for l in open(DATA / "pages.jsonl", encoding="utf-8")]
    todo = [p for p in pages if p["render"] and not (out_dir / f"p{p['page']:03d}.json").exists()]
    print(f"{len(todo)} pages to describe")
    client = OpenAI()
    errors = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(describe_page, client, p): p["page"] for p in todo}
        for f in as_completed(futs):
            pno = futs[f]
            try:
                r = f.result()
                (out_dir / f"p{pno:03d}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception as e:  # logged; page keeps its text/table chunks
                errors.append({"page": pno, "error": f"{type(e).__name__}: {e}"})
    done = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("p*.json"))]
    for d in done:  # normalize older cached outputs too
        d["elements"] = [e if isinstance(e, dict) else {"kind": "image", "label": None, "details": str(e)} for e in d["elements"]]
    summary = {"model": VISION_MODEL, "pages_described": len(done),
               "elements": sum(len(d["elements"]) for d in done),
               "by_kind": {k: sum(e.get("kind") == k for d in done for e in d["elements"])
                           for k in ("chart", "diagram", "image", "table", "decorative")},
               "prompt_tokens": sum(d["usage"]["prompt_tokens"] for d in done),
               "completion_tokens": sum(d["usage"]["completion_tokens"] for d in done),
               "cost_usd": round(sum(d["cost_usd"] for d in done), 4), "errors": errors}
    (DATA / "describe_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
