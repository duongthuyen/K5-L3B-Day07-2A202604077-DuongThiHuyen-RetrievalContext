# Cấu hình nâng cao đã chốt (trước khi chạy test, 2026-11-16)

Tiêu chí chọn, chỉ dùng dev: (1) tỷ lệ đạt trên dev; (2) nếu hòa, chọn cấu hình ít phải dựa vào mô tả do model sinh ra mà không kiểm tra ảnh gốc (ADVANCED.md: "mô tả do model tạo không tự bảo đảm đúng"); (3) chi phí/thời gian.

| Cấu hình | Dev đạt | Chi phí dev | Median/câu | Ghi chú |
|---|---|---|---|---|
| V0_text_dense | 19/21 | $0.033 | 2.6 s | chỉ có text |
| V1_mm_hybrid | 19/21 | $0.020 | 2.2 s | chạy trước bản sửa kind-group |
| V2_mm_hybrid_view | 19/21 | $0.033 | 2.2 s | chạy trước bản sửa kind-group |
| V3_mm_hybrid_view_fix | 21/21 | $0.027 | 2.2 s | 0 lần xem trang |
| **V4_mm_hybrid_mustview** | **21/21** | $0.065 | 3.5 s | xem trang gốc khi dựa vào hình/bảng (trung bình 1 trang/câu) |

**Chốt: V4_mm_hybrid_mustview**, gồm:
- Retrieval: hybrid RRF (dense bge-small trên pgvector `adv_chunks` + BM25, mỗi bên 30 candidate) → top-5, cộng tối đa 2 chunk bảng/hình cùng trang.
- Agent: `gpt-4.1-mini`, temperature 0, tối đa 4 search, tối đa 3 lần xem trang (trang đã xuất hiện trong kết quả hoặc trang kề ±1), budget 24 000 ký tự, sửa JSON khi model trả lời dạng văn xuôi.

Test (30 câu) chỉ chạy một lần với cấu hình này, kèm V0 làm baseline để so sánh. Không chỉnh gì sau khi chạy test.
