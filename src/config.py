"""Shared settings for Day 7. Everything identical across experiments lives here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DAY6_DIR = ROOT.parent / "K5-L3-Day06-DuongThiHuyen-2A202604077-DataIngestion"
DATA_DIR = ROOT / "data"
BENCH_DIR = ROOT / "benchmark"
RESULTS_DIR = ROOT / "results"

CHUNKS_PATH = DATA_DIR / "chunks.json"          # exported from pgvector (strategy B)
QUESTIONS_PATH = DATA_DIR / "questions.json"    # answerable, Day 7 schema
CASES_PATH = DATA_DIR / "cases.json"            # all cases (answerable + unanswerable)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
PG_STRATEGY = "B"   # Day 6 chosen chunking (paragraph <=1000 chars); Day 6 config C = B + filter

# Must match the Day 6 index exactly.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
# USD per 1M tokens (OpenAI list price for gpt-4o-mini; verify before reuse).
PRICE_PER_M = {"gpt-4o-mini": (0.15, 0.60)}

FILTERABLE_FIELDS = {"book": str, "chapter": int, "document_id": str}
