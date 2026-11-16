"""Retrievers over the advanced index (adv_chunks): dense pgvector, BM25, hybrid RRF, + same-page
neighbours. Reuses the Day 7 implementations; only the dense SQL differs (different table)."""
from __future__ import annotations

import json
from typing import Any

import psycopg

from src.retrievers import BM25Retriever, HybridRRF, _embedder
from src.config import QUERY_INSTRUCTION

from .common import DATA, DATABASE_URL, TABLE

KINDS = {"text", "table", "visual"}
# Dev fix (AD01): a table may exist only as a vision-described element (find_tables missed it), so
# "table" and "visual" both mean "non-text" chunks.
KIND_GROUPS = {"text": ["text"], "table": ["table", "visual"], "visual": ["table", "visual"]}


def load_chunks() -> list[dict[str, Any]]:
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    for c in chunks:  # adapt to the Day 7 retriever schema
        c["source_id"], c["source"] = "openstax-data-science-full", f"data-science.pdf#page={c['page']}"
        c["metadata"] = {"kind": c["kind"], "label": c.get("label"), "section": c.get("section")}
    return chunks


class AdvDense:
    name = "dense"

    def __init__(self, chunks):
        self.by_id = {c["id"]: c for c in chunks}

    def search(self, query: str, *, filters=None, top_k: int = 5):
        vec = _embedder().encode(QUERY_INSTRUCTION + query, normalize_embeddings=True, show_progress_bar=False)
        lit = "[" + ",".join(f"{x:.7f}" for x in vec.tolist()) + "]"
        where, params = "", []
        if filters and filters.get("kind") in KINDS:
            where, params = "WHERE kind = ANY(%s)", [KIND_GROUPS[filters["kind"]]]
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            rows = conn.execute(f"SELECT id, 1 - (embedding <=> %s::vector) FROM {TABLE} {where} "
                                f"ORDER BY embedding <=> %s::vector, id LIMIT %s", [lit, *params, lit, top_k]).fetchall()
        return [{**self.by_id[r[0]], "score": float(r[1]), "rank": i + 1, "strategy": self.name} for i, r in enumerate(rows)]


class KindBM25(BM25Retriever):
    def search(self, query, *, filters=None, top_k=5):
        res = super().search(query, top_k=top_k * 4 if filters else top_k)
        if filters and filters.get("kind") in KINDS:
            res = [r for r in res if r["kind"] in KIND_GROUPS[filters["kind"]]][:top_k]
            res = [{**r, "rank": i + 1} for i, r in enumerate(res)]
        return res


class SamePage:
    """Add the other chunks of the same page that are visual/table chunks (a chart's description
    usually sits next to the text that discusses it) — capped to keep context small."""

    def __init__(self, inner, chunks, max_extra: int = 2):
        self.inner, self.max_extra = inner, max_extra
        self.by_page: dict[int, list] = {}
        for c in chunks:
            self.by_page.setdefault(c["page"], []).append(c)
        self.name = inner.name + "+samepage"

    def search(self, query, *, filters=None, top_k=5):
        res = self.inner.search(query, filters=filters, top_k=top_k)
        seen = {r["id"] for r in res}
        extra = []
        for r in res:
            sib = [c for c in self.by_page[r["page"]] if c["id"] not in seen and c["kind"] != "text"]
            for c in sib[: self.max_extra]:
                extra.append({**c, "score": None, "rank": None, "strategy": "same_page"}); seen.add(c["id"])
        return res + extra


def build(method: str, chunks):
    dense, bm25 = AdvDense(chunks), KindBM25(chunks)
    base = {"dense": dense, "bm25": bm25, "hybrid": HybridRRF(dense, bm25, candidates=30)}
    if method.endswith("+samepage"):
        return SamePage(base[method.removesuffix("+samepage")], chunks)
    return base[method]
