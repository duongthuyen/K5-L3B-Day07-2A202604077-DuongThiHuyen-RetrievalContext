"""Retrieval methods compared in Day 7. All return the same result schema:
{id, text, source_id, source, page, metadata, score, rank, strategy}.

1. DenseRetriever   - bge-small-en-v1.5 query embedding -> Day 6 pgvector (cosine, exact)
2. BM25Retriever    - lexical BM25 (Okapi, k1=1.5, b=0.75) over the same chunks
3. HybridRRF        - Reciprocal Rank Fusion of dense + BM25 candidate lists (k=60)
4. RerankRetriever  - any candidate retriever + cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
Filters only come from user-supplied context (question['filter']), never from gold.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any

import psycopg
from psycopg import sql

from .config import (DATABASE_URL, EMBEDDING_DIM, EMBEDDING_MODEL, FILTERABLE_FIELDS, PG_STRATEGY,
                     QUERY_INSTRUCTION, RERANK_MODEL)

Filters = dict[str, Any] | None


def _check_filters(filters: Filters) -> dict[str, Any]:
    out = {}
    for k, v in (filters or {}).items():
        if k not in FILTERABLE_FIELDS:
            raise ValueError(f"filter {k!r} not allowed; allowed {sorted(FILTERABLE_FIELDS)}")
        out[k] = FILTERABLE_FIELDS[k](v)
    return out


def _match(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    for k, v in filters.items():
        val = chunk["source_id"] if k == "document_id" else chunk["metadata"].get(k)
        if val != v:
            return False
    return True


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    if m.get_sentence_embedding_dimension() != EMBEDDING_DIM:
        raise RuntimeError("embedding dimension mismatch with Day 6 index")
    return m


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device="cpu")


class DenseRetriever:
    name = "dense"

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url

    def search(self, query: str, *, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        f = _check_filters(filters)
        vec = _embedder().encode(QUERY_INSTRUCTION + query, normalize_embeddings=True, show_progress_bar=False)
        lit = "[" + ",".join(f"{x:.7f}" for x in vec.tolist()) + "]"
        where, params = [sql.SQL("strategy = %s")], [PG_STRATEGY]
        for k, v in f.items():
            where.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
            params.append(v)
        q = sql.SQL("""SELECT chunk_id, document_id, source, page, content, metadata,
                              1 - (embedding <=> %s::vector) AS score
                       FROM chunks WHERE {w} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"""
                    ).format(w=sql.SQL(" AND ").join(where))
        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            rows = conn.execute(q, [lit, *params, lit, top_k]).fetchall()
        return [{"id": r[0], "source_id": r[1], "source": r[2], "page": r[3], "text": r[4], "metadata": r[5],
                 "score": float(r[6]), "rank": i + 1, "strategy": self.name} for i, r in enumerate(rows)]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        f = _check_filters(filters)
        q = [t for t in tokenize(query) if t in self.idf]
        scored = []
        for c, tf, dl in zip(self.chunks, self.tfs, self.lens):
            if f and not _match(c, f):
                continue
            s = 0.0
            for t in q:
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [{**c, "score": s, "rank": i + 1, "strategy": self.name} for i, (s, c) in enumerate(scored[:top_k])]


class HybridRRF:
    """score(d) = sum_r 1/(k + rank_r(d)); raw scores are never added across retrievers."""
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, candidates: int = 20, k: int = 60):
        self.dense, self.bm25, self.candidates, self.k = dense, bm25, candidates, k

    def search(self, query: str, *, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for r in (self.dense, self.bm25):
            for item in r.search(query, filters=filters, top_k=self.candidates):
                e = fused.setdefault(item["id"], {**item, "score": 0.0, "components": {}})
                e["score"] += 1 / (self.k + item["rank"])
                e["components"][r.name] = {"rank": item["rank"], "score": item["score"]}
        ranked = sorted(fused.values(), key=lambda x: (-x["score"], x["id"]))[:top_k]
        return [{**c, "rank": i + 1, "strategy": self.name} for i, c in enumerate(ranked)]


class RerankRetriever:
    """Take `candidates` from a first-stage retriever, rescore (query, chunk) with a cross-encoder."""

    def __init__(self, first_stage, candidates: int = 20):
        self.first, self.candidates = first_stage, candidates
        self.name = f"{first_stage.name}+rerank"

    def search(self, query: str, *, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        cands = self.first.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = _cross_encoder().predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        for c, s in zip(cands, scores):
            c["first_stage_rank"], c["first_stage_score"], c["score"] = c["rank"], c["score"], float(s)
        cands.sort(key=lambda c: (-c["score"], c["id"]))
        return [{**c, "rank": i + 1, "strategy": self.name} for i, c in enumerate(cands[:top_k])]


class NeighbourRetriever:
    """Wrap a retriever: top_k results + prev/next chunk of the same page (marked strategy='neighbour').
    Returns more than top_k items by design; ranking metrics use only the ranked ones."""

    def __init__(self, inner, chunks: list[dict[str, Any]]):
        from .context import expand_neighbours
        self.inner, self._expand = inner, expand_neighbours
        self.by_id = {c["id"]: c for c in chunks}
        self.ordered = [c["id"] for c in chunks]
        self.name = f"{inner.name}+neighbours"

    def search(self, query: str, *, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self._expand(self.inner.search(query, filters=filters, top_k=top_k), self.by_id, self.ordered)


def build(method: str, chunks: list[dict[str, Any]], candidates: int = 20):
    if method.endswith("+neighbours"):
        return NeighbourRetriever(build(method.removesuffix("+neighbours"), chunks, candidates), chunks)
    dense = DenseRetriever()
    bm25 = BM25Retriever(chunks)
    return {
        "dense": lambda: dense,
        "bm25": lambda: bm25,
        "hybrid": lambda: HybridRRF(dense, bm25, candidates=candidates),
        "dense_rerank": lambda: RerankRetriever(dense, candidates=candidates),
        "hybrid_rerank": lambda: RerankRetriever(HybridRRF(dense, bm25, candidates=candidates), candidates=candidates),
    }[method]()
