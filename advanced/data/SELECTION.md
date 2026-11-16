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

**Kết quả round 1:** test V4 đạt 25/30 (83.3%), V0 đạt 15/30 (50%); nhãn ở `results/advanced/reviews/*-test.json`. Chưa đạt ngưỡng 90%, và vì phải đọc test để phân tích lỗi, nên theo ADVANCED.md test này chuyển thành **dev2** và một **test2 mới** được tạo ([benchmark_round2_test.json](benchmark_round2_test.json), 30 câu: 6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text, 6 thiếu evidence; commit 92a4135 trước mọi thay đổi của round 2).

# Round 2: chốt trước khi chạy test2

Tiêu chí như round 1, áp dụng trên dev + dev2.

| Cấu hình | Dev | Dev2 (test cũ) | Chi phí dev + dev2 | Median/câu |
|---|---|---|---|---|
| V4 (round 1) | 21/21 | 25/30 | $0.065 + $0.105 | 3.5 s |
| V5_round2_gpt41 | 21/21 | 25/30 | $0.31 + $0.49 | 3.1-3.5 s |
| **V6_round2_zoom_gpt5mini** | **21/21** | **29/30** | $0.11 + $0.15 | 14 s |

**Chốt: V6_round2_zoom_gpt5mini.** So với V4, V6 thêm:
- Model `gpt-5-mini` (reasoning effort medium) cho agent.
- `view_page` gửi thêm crop 220 dpi của tối đa 3 vùng hình/bảng lớn nhất trên trang.
- Chunk của trang đã xem được phép trích dẫn.
- Quy tắc đọc hình: lần theo từng cạnh, bin dạng [mép trái, mép phải], kiểm tra trường hợp hòa.
- Quy tắc từ chối: chỉ nêu phần còn thiếu.
- Parser lấy ID citation bằng regex.

Đánh đổi: chậm hơn khoảng 4 lần (median 14 s/câu) nhưng rẻ hơn V5. Lỗi còn lại trên dev2 là AT14: đọc giá trị nhỏ nhất của một biểu đồ đường dày đặc.

test2 chỉ chạy một lần với V6, kèm V0 làm baseline.
