"""Shared paths/settings for the advanced (100+ page multimodal) challenge."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ADV = ROOT / "advanced"
DATA = ADV / "data"            # committed: extracted text/tables/descriptions, benchmark, chunks
CACHE = ADV / "cache"          # git-ignored: page renders and crops (re-creatable)
RESULTS = ROOT / "results" / "advanced"

# OpenStax "Principles of Data Science" (2024 web edition), CC BY-NC-SA 4.0. Same file as Day 6
# (downloaded by Day 6 scripts/download_corpus.py, hash-checked).
PDF_PATH = ROOT.parent / "K5-L3-Day06-DuongThiHuyen-2A202604077-DataIngestion" / "starter" / "data" / "raw" / "data-science.pdf"
PDF_URL = "https://assets.openstax.org/oscms-prodcms/media/documents/Principles-of-Data-Science-WEB.pdf"
PDF_SHA256 = "6b47205459a2f4e25a2b2a86369e674a26a5ae566a4cc428dd31047670692398"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
TABLE = "adv_chunks"

VISION_MODEL = os.getenv("ADV_VISION_MODEL", "gpt-4.1-mini")
AGENT_MODEL = os.getenv("ADV_AGENT_MODEL", "gpt-4.1-mini")
# USD per 1M tokens (input, output) — OpenAI list prices; verify before reuse.
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4o-mini": (0.15, 0.60)}


def cost(model: str, p: int, c: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return p / 1e6 * pin + c / 1e6 * pout
