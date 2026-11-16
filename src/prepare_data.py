"""Export Day 6 chunks from pgvector and build the Day 7 benchmark files.

- data/chunks.json   : every chunk of index B read from the live Day 6 vector store
- data/questions.json: answerable questions (Day 6 q01-q16 + t01-t06 -> Day 7 dev,
                       new n01-n08 -> Day 7 test), gold = list of alternative evidence
- data/cases.json    : all cases incl. unanswerable (for agent metrics)
Validates that every gold evidence lies inside one chunk of the right source/page.
Run: python -m src.prepare_data
"""
from __future__ import annotations

import json
from collections import Counter

import psycopg

from .config import BENCH_DIR, CASES_PATH, CHUNKS_PATH, DATABASE_URL, DAY6_DIR, EMBEDDING_MODEL, PG_STRATEGY, QUESTIONS_PATH
from .data import is_hit, load_json, save_json


def export_chunks() -> list[dict]:
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
        rows = conn.execute(
            """SELECT chunk_id, document_id, source, page, content, metadata, embedding_model,
                      vector_dims(embedding)
               FROM chunks WHERE strategy=%s ORDER BY document_id, page, chunk_id""", (PG_STRATEGY,)).fetchall()
    if not rows:
        raise RuntimeError(f"No chunks for strategy {PG_STRATEGY}: start Day 6 docker and run its ingest")
    chunks = []
    for cid, doc, src, page, text, meta, model, dims in rows:
        if model != EMBEDDING_MODEL or dims != 384:
            raise RuntimeError(f"{cid}: index built with {model}/{dims}, expected {EMBEDDING_MODEL}/384")
        chunks.append({"id": cid, "text": text, "source_id": doc, "source": src, "page": page, "metadata": meta})
    return chunks


def convert_day6_questions() -> list[dict]:
    out = []
    for line in (DAY6_DIR / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
        gold += [{"source_id": a["gold_source_id"], "page": a["gold_page"], "evidence": a["evidence_quote"]}
                 for a in q.get("alternative_evidence", [])]
        item = {"id": q["id"], "split": "dev",  # Day 6 test was already seen -> only used as Day 7 dev
                "day6_split": q["split"], "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "gold": gold}
        if "filter" in q:
            item["filter"] = q["filter"]
        out.append(item)
    return out


def main() -> None:
    chunks = export_chunks()
    test2 = load_json(BENCH_DIR / "test2.json")  # created after test was viewed (citation-id fix)
    questions = (convert_day6_questions() + load_json(BENCH_DIR / "new_test_and_unanswerable.json")
                 + [q for q in test2 if q.get("answerable", True)])
    unanswerable = load_json(BENCH_DIR / "unanswerable.json") + [q for q in test2 if q.get("answerable") is False]
    for q in questions:
        for g in q["gold"]:
            if not any(is_hit(c, [g]) for c in chunks):
                raise ValueError(f"{q['id']}: evidence not inside any chunk of {g['source_id']} p.{g['page']}")
    ids = [q["id"] for q in questions + unanswerable]
    assert len(ids) == len(set(ids)), "duplicate ids"
    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
              "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if "filter" in q else {})}
             for q in questions] + unanswerable
    save_json(CHUNKS_PATH, chunks)
    save_json(QUESTIONS_PATH, questions)
    save_json(CASES_PATH, cases)
    print(f"chunks={len(chunks)} pages={len({(c['source_id'], c['page']) for c in chunks})}")
    print("cases per split/answerable:", dict(Counter((c["split"], c["answerable"]) for c in cases)))


if __name__ == "__main__":
    main()
