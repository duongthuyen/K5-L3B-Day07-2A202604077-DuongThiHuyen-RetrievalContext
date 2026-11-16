# Benchmark Day 7 — 36 câu (dev 22 + 3 thiếu evidence · test 8 + 3 thiếu evidence)

Corpus: 56 trang, 189 chunk (index **B** của Day 6, đọc trực tiếp từ pgvector, xuất ra [data/chunks.json](../data/chunks.json)).
Build: `python -m src.prepare_data` → [data/questions.json](../data/questions.json), [data/cases.json](../data/cases.json).

| File | Nội dung |
|---|---|
| Day 6 `benchmark/questions.jsonl` | q01–q16 (dev Day 6) + t01–t06 (test Day 6) → **đều thành dev Day 7** vì test Day 6 đã được xem ở Day 6 (`day6_split` giữ nguồn gốc) |
| [new_test_and_unanswerable.json](new_test_and_unanswerable.json) | **n01–n08: test mới của Day 7**, viết từ các trang chưa từng là gold của dev (89, 93, 95, 97, 101, 230, 239), viết và commit trước khi chạy bất kỳ thí nghiệm nào |
| [unanswerable.json](unanswerable.json) | u01–u03 (dev), u04–u06 (test): câu gần chủ đề nhưng thiếu evidence; `absence_reason` ghi cách kiểm tra (grep toàn bộ 56 trang đã chuẩn hóa, không chỉ top-k) |

Schema câu có đáp án: `id, split, type, question, expected_answer, gold[{source_id, page, evidence}]`, `filter` tùy chọn. Các phần tử `gold` là evidence **thay thế** (một là đủ). Mọi câu đều single-evidence; không có câu multi-hop.

**Định nghĩa hit** (dùng chung mọi cấu hình, [src/data.py](../src/data.py)): chunk cùng `source_id`, cùng `page` và chứa trọn đoạn evidence sau khi chuẩn hóa khoảng trắng (PDF ngắt dòng giữa câu / ô bảng).

**Filter**: chỉ từ trường `filter` = ngữ cảnh người dùng cung cấp trước (q11, q12, q13, t01, t02); không suy ra từ gold. Gold/expected_answer không bao giờ vào retriever hay agent (gold chỉ dùng để chấm sau khi chạy).

## Tiêu chí chọn cấu hình (chỉ dùng dev)

1. Context Evidence Rate dev cao nhất → 2. Evidence Hit@3 → 3. Hit@1 → 4. median latency / chi phí thấp hơn.
Với agent: Grounded answer trung bình, rồi từ chối nhầm thấp hơn, rồi chi phí.

> Ghi chú trung thực: tiêu chí này được viết ra sau lượt chạy dev đầu tiên (R0–R6), trước khi thêm R7/R8 và **trước mọi lần chạy test**.

## Cấu hình đã chốt (trước khi chạy test, 2026-11-16)

- Retrieval: **R8_hybrid_neighbours** — RRF(dense pgvector top-20, BM25 top-20, k=60) → top-3 + chunk trước/sau cùng trang, budget 6000 ký tự, filter người dùng.
- Agent: **A1_hybrid_neighbours** — gpt-4o-mini, temperature 0, tool `search_corpus` dùng R8 (top-3 + neighbours), tối đa 3 search, budget 8000 ký tự.
- Baseline để so sánh trên test: R0_dense / A0_dense (Day 6 config C).

Test chỉ chạy một lần cho baseline + cấu hình đã chọn; không chỉnh tham số sau đó.
