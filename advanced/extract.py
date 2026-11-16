"""Step 1 — extract every page of the 569-page OpenStax *Principles of Data Science* PDF.

Per page (PyMuPDF, born-digital PDF -> text layer is the original text, no OCR needed):
- text blocks (reading order), printed page number, chapter/section
- tables via page.find_tables() -> markdown + bbox
- visual regions: embedded raster images + clusters of vector drawings (charts/diagrams)
- captions "Figure x.y ..." / "Table x.y ..." with bbox
- page render (PNG, 110 dpi) for pages that have visuals/tables (used by the vision step and
  by the agent's view_page tool), plus a crop per visual region for verification.
Output: advanced/data/pages.jsonl ; skipped pages + reason in advanced/data/extract_report.json
Run: python -m advanced.extract
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from .common import ADV, CACHE, DATA, PDF_PATH, PDF_SHA256

MIN_CHARS = 200          # page counted as content if >= this many chars of text or has a visual
RENDER_DPI = 110
CAP_RE = re.compile(r"^(Figure|Table)\s+(\d+\.\d+)\s+(.*)")


def _cluster_drawings(page) -> list[pymupdf.Rect]:
    """Union nearby vector-drawing rects into regions; keep regions big enough to be a figure."""
    rects = [d["rect"] for d in page.get_drawings() if d["rect"].width < page.rect.width * 0.95
             or d["rect"].height < page.rect.height * 0.5]
    regions: list[pymupdf.Rect] = []
    for r in rects:
        if r.is_empty and r.width == 0 and r.height == 0:
            continue
        r = pymupdf.Rect(r)
        merged = True
        while merged:
            merged = False
            for g in regions:
                if (g + (-8, -8, 8, 8)).intersects(r):
                    r |= g
                    regions.remove(g)
                    merged = True
                    break
        regions.append(r)
    return [g for g in regions if g.width > 120 and g.height > 80]


def extract() -> dict:
    doc = pymupdf.open(PDF_PATH)
    sha = hashlib.sha256(Path(PDF_PATH).read_bytes()).hexdigest()
    if sha != PDF_SHA256:
        raise RuntimeError(f"PDF hash mismatch: {sha}")
    toc = doc.get_toc()
    out, skipped = [], []
    (CACHE / "pages").mkdir(parents=True, exist_ok=True)
    (CACHE / "crops").mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        pno = i + 1
        chapter = section = None
        for lvl, title, start in toc:
            if start <= pno:
                if lvl == 1:
                    chapter, section = title, None
                elif lvl == 2:
                    section = title
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        text = "\n\n".join(b[4].strip() for b in blocks if b[4].strip())
        m = re.search(r"^(\d{1,3})\s", text) or re.search(r"\s(\d{1,3})$", text.split("\n")[0] if text else "")
        printed = int(m.group(1)) if m else None
        caps = []
        for b in blocks:
            cm = CAP_RE.match(b[4].strip().replace("\n", " "))
            if cm:
                caps.append({"kind": cm.group(1).lower(), "label": f"{cm.group(1)} {cm.group(2)}",
                             "caption": cm.group(3)[:300], "bbox": [round(x, 1) for x in b[:4]]})
        tables = []
        try:
            for t in page.find_tables().tables:
                md = t.to_markdown()
                if md and len(md) > 40:
                    tables.append({"bbox": [round(x, 1) for x in t.bbox], "markdown": md})
        except Exception as e:  # recorded, not hidden
            tables.append({"error": f"{type(e).__name__}: {e}"})
        visuals = []
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"])
            if r.width > 60 and r.height > 40:
                visuals.append({"type": "raster", "bbox": [round(x, 1) for x in r]})
        tbl_rects = [pymupdf.Rect(t["bbox"]) for t in tables if "bbox" in t]
        for r in _cluster_drawings(page):
            if any(r.intersects(tr) and (r & tr).get_area() > 0.6 * r.get_area() for tr in tbl_rects):
                continue  # table borders, already captured as a table
            visuals.append({"type": "vector", "bbox": [round(x, 1) for x in r]})
        has_visual = bool(visuals or tables)
        if len(text) < MIN_CHARS and not has_visual:
            skipped.append({"page": pno, "chars": len(text), "reason": "blank/near-empty page"})
            continue
        render = None
        if has_visual or caps:
            render = f"pages/p{pno:03d}.png"
            page.get_pixmap(dpi=RENDER_DPI).save(CACHE / render)
            for k, v in enumerate(visuals):
                clip = pymupdf.Rect(v["bbox"]) + (-10, -10, 10, 30)
                v["crop"] = f"crops/p{pno:03d}_v{k}.png"
                page.get_pixmap(dpi=RENDER_DPI, clip=clip & page.rect).save(CACHE / v["crop"])
        out.append({"page": pno, "printed_page": printed, "chapter": chapter, "section": section,
                    "text": text, "captions": caps, "tables": tables, "visuals": visuals, "render": render})
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    report = {"pdf": str(PDF_PATH.name), "sha256": sha, "total_pages": len(doc), "indexed_pages": len(out),
              "skipped": skipped,
              "pages_with_tables": sum(bool(p["tables"]) for p in out),
              "pages_with_visuals": sum(bool(p["visuals"]) for p in out),
              "tables": sum(len(p["tables"]) for p in out), "visual_regions": sum(len(p["visuals"]) for p in out),
              "captions": {k: sum(c["kind"] == k for p in out for c in p["captions"]) for k in ("figure", "table")}}
    (DATA / "extract_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = extract()
    print({k: v for k, v in r.items() if k != "skipped"}, "skipped:", len(r["skipped"]))
