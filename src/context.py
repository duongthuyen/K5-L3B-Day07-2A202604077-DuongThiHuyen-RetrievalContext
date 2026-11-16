"""Context selection: dedup by chunk id, keep relevance order, optional same-page
neighbour expansion, hard character budget (headers included)."""
from __future__ import annotations

from typing import Any


def block(c: dict[str, Any]) -> str:
    sec = c["metadata"].get("section") or ""
    return f"[{c['id']}] (source: {c['source']}, page {c['page']}, section: {sec})\n{c['text']}"


def expand_neighbours(results: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, Any]],
                      ordered_ids: list[str]) -> list[dict[str, Any]]:
    """After each retrieved chunk, add its previous and next chunk on the SAME page
    (dev error analysis: evidence often sits in the adjacent chunk of the right page).
    Retrieved chunks keep priority; neighbours are appended after all retrieved chunks."""
    pos = {cid: i for i, cid in enumerate(ordered_ids)}
    out = list({r["id"]: r for r in results}.values())
    seen = {r["id"] for r in out}
    for r in results:
        i = pos.get(r["id"])
        if i is None:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(ordered_ids):
                nb = chunks_by_id[ordered_ids[j]]
                if nb["source_id"] == r["source_id"] and nb["page"] == r["page"] and nb["id"] not in seen:
                    out.append({**nb, "score": None, "rank": None, "strategy": "neighbour"}); seen.add(nb["id"])
    return out


def pack(results: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    blocks, ids, used = [], [], 0
    for c in results:
        if c["id"] in ids:
            continue
        b = block(c)
        extra = len(b) + (2 if blocks else 0)
        if used + extra > budget:
            continue  # skip chunks that do not fit; smaller later chunks may still fit
        blocks.append(b); ids.append(c["id"]); used += extra
    return "\n\n".join(blocks), ids
