"""Load corpus/benchmark and the single shared definition of an evidence hit."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import CASES_PATH, CHUNKS_PATH, QUESTIONS_PATH


def norm(text: str) -> str:
    """NFC + collapse whitespace. PDF text has hard line breaks inside sentences/table
    cells, so evidence quotes are compared after whitespace normalization."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Hit = same source_id AND same page AND normalized evidence fully inside the chunk.
    Elements of `gold` are alternative valid evidence (any one suffices)."""
    t = norm(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and int(chunk["page"]) == int(g["page"])
               and norm(g["evidence"]) in t for g in gold)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_chunks() -> list[dict[str, Any]]:
    return load_json(CHUNKS_PATH)


def load_questions(split: str) -> list[dict[str, Any]]:
    return [q for q in load_json(QUESTIONS_PATH) if q["split"] == split]


def load_cases(split: str) -> list[dict[str, Any]]:
    return [c for c in load_json(CASES_PATH) if c["split"] == split]
