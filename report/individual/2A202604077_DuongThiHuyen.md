# Báo cáo Day 7 — Agentic RAG · Dương Thị Huyền (2A202604077)

Cách chạy lại: xem [README.md](../../README.md#bài-làm-cách-chạy-lại). Benchmark và cấu hình đã chốt: [benchmark/README.md](../../benchmark/README.md).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** dùng lại corpus Day 6 — 2 giáo trình OpenStax (CC BY-NC-SA 4.0, Rice University): *Introduction to Python Programming* ch. 9 Lists + ch. 10 Dictionaries (PDF tr. 221–258) và *Principles of Data Science* mục 2.4–2.5 (tr. 86–103). **56 trang nội dung, 189 chunk** (index B của Day 6: chunk theo heading/đoạn, ≤1000 ký tự). Chunk được đọc thẳng từ pgvector và xuất ra [data/chunks.json](../../data/chunks.json) (`python -m src.prepare_data`, có kiểm tra model + số chiều của từng dòng).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker, Day 6 `docker compose -p day6_student_starter`), bảng `chunks`, `strategy='B'`, tìm chính xác bằng cosine `<=>` (không ANN). Embedding `BAAI/bge-small-en-v1.5`, **384 chiều**, chuẩn hóa L2; query thêm instruction bge `"Represent this sentence for searching relevant passages: "` đúng như lúc Day 6 tạo index. Không tạo lại index, nên không tốn chi phí embedding.
- **Agent model, cách chạy và phân chia dev/test:** `gpt-4o-mini` (OpenAI Chat Completions, tool calling, temperature 0). Reranker dùng thử: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU). Dev = 22 câu có đáp án (q01–q16 và t01–t06 của Day 6; **test cũ của Day 6 chuyển sang dev** vì đã được xem ở Day 6) + 3 câu thiếu evidence (u01–u03). Test = **8 câu mới** (n01–n08), viết từ các trang chưa từng là gold của dev, + 3 câu thiếu evidence (u04–u06). Test được viết trước mọi thí nghiệm và chỉ chạy một lần sau khi đã chốt cấu hình.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold = [{source_id, page, evidence}]`, các phần tử là evidence thay thế. **Hit** = chunk cùng nguồn, cùng trang và chứa trọn evidence sau khi chuẩn hóa khoảng trắng ([src/data.py](../../src/data.py)); cả 30 câu đều được kiểm tra tự động là có chunk khớp. Câu thiếu evidence là câu gần chủ đề (`reverse=True`, pandas, độ phức tạp, KNN, phiên bản Python, deepcopy); với mỗi câu, mình grep toàn bộ 56 trang đã chuẩn hóa chứ không chỉ xem top-k ([benchmark/unanswerable.json](../../benchmark/unanswerable.json)). Câu thiếu evidence không nằm trong mẫu số của các metric retrieval. Filter chỉ lấy từ trường `filter` mà người dùng cung cấp trước (q11–q13, t01, t02); gold không bao giờ được đưa vào retriever hay agent.

## Retrieval và thử nghiệm

Cùng 189 chunk, cùng index pgvector B, cùng câu hỏi và nhãn. Query gửi đi = câu hỏi gốc. Hit@1 và Evidence Hit@3 luôn tính trên ranking top-3. Context = các chunk được đóng gói theo budget ký tự (gồm cả header nguồn/trang). Thời gian là median mỗi query trên CPU laptop. Chi phí retrieval là $0, vì embedding và reranker chạy local.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline R0_dense | Dense pgvector, top-3, filter người dùng, không rerank, 3000 ký tự | Config C của Day 6 | dev | 14/22 (0.64) | 19/22 (0.86) | 19/22 (0.86) | 102 ms / $0 |
| Phương pháp 2 R1_bm25 | BM25 (k1 = 1.5, b = 0.75, token `\w+`), top-3, filter, 3000 ký tự | Tên hàm như `sum()` hay `pop()` khớp chính xác | dev | 16/22 (0.73) | 21/22 (0.95) | 21/22 (0.95) | 0.5 ms / $0 |
| Phương pháp 3 / Cải thiện 1 R2_hybrid | RRF(dense top-20, BM25 top-20, k = 60) → top-3, filter, 3000 ký tự | Hai retriever bù lỗi cho nhau | dev | 16/22 (0.73) | 21/22 (0.95) | 21/22 (0.95) | 93 ms / $0 |
| Phương pháp 4 / Cải thiện 2 R3_hybrid_rerank | RRF top-20 → cross-encoder → top-3, 3000 ký tự | Rerank sửa thứ tự, tăng Hit@1 | dev | 16/22 (0.73) | 20/22 (0.91) | 20/22 (0.91) | 943 ms / $0 |
| Ablation R4_dense_rerank | Dense top-20 → cross-encoder → top-3 | Rerank khi không có lexical | dev | 16/22 (0.73) | 20/22 (0.91) | 20/22 (0.91) | 2082 ms / $0 |
| Ablation R5_hybrid_rerank_k5 | Như R3, context top-5, 4500 ký tự | Thêm context cho evidence ở hạng 4–5 | dev | 16/22 (0.73) | 20/22 (0.91) | 20/22 (0.91) | 2020 ms / $0 |
| Ablation R6_dense_nofilter | Dense, không filter (config B của Day 6) | Đo tác dụng của filter | dev | 14/22 (0.64) | 19/22 (0.86) | 19/22 (0.86) | 55 ms / $0 |
| Cải thiện 3 R7_dense_neighbours | Dense top-3 + chunk trước/sau cùng trang, 6000 ký tự | Lỗi dev: đúng trang nhưng evidence nằm ở chunk kề bên | dev | 14/22 (0.64) | 19/22 (0.86) | 21/22 (0.95) | 55 ms / $0 |
| **Cải thiện 4 R8_hybrid_neighbours** | RRF top-3 + chunk kề cùng trang, 6000 ký tự | Kết hợp R2 và R7 | dev | 16/22 (0.73) | 21/22 (0.95) | **22/22 (1.00)** | 60 ms / $0 |
| **Cấu hình đã chọn R8** | như trên | — | test | 5/8 (0.62) | 8/8 (1.00) | 8/8 (1.00) | 58 ms / $0 |
| Baseline R0_dense | như trên | — | test | 5/8 (0.62) | 7/8 (0.88) | 7/8 (0.88) | 55 ms / $0 |

Context trung bình trên dev: R0 2047 ký tự, R8 3927 ký tự (≈ 1000 token); trên test: R0 2263, R8 4494. Trace từng query nằm ở `results/retrieval/<config>-<split>/traces.json`, bảng tổng hợp ở `results/retrieval/comparison-dev.json` và `comparison-test.json`.

**Chấm context thủ công trên test** (8 câu, [context_labels_test.json](../../results/reviews/context_labels_test.json)): R0 có 7 câu mức 2 và 1 câu mức 0. Câu mức 0 là n05: ô bảng "Stopwords removal" (p0095:c01) không được lấy; dense chỉ lấy được p0095:c00, phần về tokenization. R8 có 8/8 câu mức 2; BM25 đưa được p0095:c01 lên hạng 3 nhờ khớp từ "stopwords".

**Hai lỗi dev, nguyên nhân, cách thử và kết quả:**

1. **Đúng trang nhưng sai chunk (q09, q13, t02 với dense).** Ví dụ q09: top-3 của dense đều là p0244, còn câu "a dictionary is a mutable data type" nằm ở p0243:c01. Với q13 và t02, dense tìm đúng trang 86 và 234 nhưng lấy c00/c03 trong khi evidence ở chunk kề bên. Nguyên nhân: chunk B tách theo đoạn nên câu định nghĩa và phần ví dụ rơi vào hai chunk khác nhau, còn embedding của chunk ví dụ lại giống câu hỏi hơn. Cách thử: thêm chunk trước/sau cùng trang vào context (R7). Context Evidence của dense tăng từ 19/22 lên 21/22, của hybrid tăng từ 21/22 lên 22/22. Hit@k giữ nguyên vì ranking không đổi. Context dài gấp khoảng 1,8 lần, độ trễ không đổi.
2. **Dense và BM25 trượt ở những câu khác nhau.** BM25 trượt q07: câu hỏi "requested key is not in the dictionary" khớp từ khóa với nhiều chunk. Dense trượt q09, q13, t02 vì diễn đạt khác với sách. RRF giữ được hit của cả hai: Hit@3 tăng từ 19/22 lên 21/22 và Hit@1 từ 14 lên 16. Câu còn trượt duy nhất là q09: không retriever nào đưa p0243:c01 vào top-3, nhưng với R8 chunk này vẫn vào context nhờ là chunk kề cùng trang.
3. **Giả thuyết không thành: rerank.** Cross-encoder MS MARCO không tăng Hit@1 (vẫn 16/22), làm Hit@3 giảm từ 21 xuống 20 (q08: chunk gold p0242:c02 bị đẩy ra khỏi top-3) và chậm hơn 10–35 lần trên CPU. Model rerank được huấn luyện cho truy vấn web, không hợp với đoạn giáo trình nhiều code. Mình ghi lại kết quả này và không chọn rerank.

**Lý do chọn:** tiêu chí được ghi trước khi chạy test ([benchmark/README.md](../../benchmark/README.md)), theo thứ tự: Context Evidence → Hit@3 → Hit@1 → latency/chi phí. R8 đứng đầu ở mọi bậc (22/22, 21/22, 16/22, 60 ms). Lưu ý trung thực: tiêu chí này được viết sau lượt dev đầu tiên (R0–R6) nhưng trước R7/R8 và trước test. Filter không đổi kết quả trên dev (R0 = R6), vì trên chunk B các câu có filter vẫn trúng mà không cần filter; mình vẫn giữ filter vì đó là ngữ cảnh người dùng cung cấp.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([src/agent.py](../../src/agent.py)):
  - Tool: một tool `search_corpus(query)`. Model tự viết query, có thể viết lại hoặc thu hẹp query. Filter người dùng được hệ thống gắn vào, model không tự đặt filter.
  - Retriever và số lượt: A0 dùng R0 (dense), A1 dùng R8 (hybrid + chunk kề). Mỗi search lấy top-3; tối đa 3 search.
  - Chọn context: chunk đã gửi rồi thì không gửi lại, chỉ báo "Already shown". Tổng context duy nhất bị giới hạn ở 8000 ký tự; chunk vượt budget bị bỏ và có ghi trong trace.
  - Điều kiện dừng: model trả JSON cuối, hoặc đã dùng hết 3 search, khi đó có thêm một lượt bắt buộc trả lời với `tool_choice="none"`.
  - Citation: phải là ID của chunk đã thực sự gửi cho model. Nếu không phải, hoặc không có citation, kết quả là abstain và lý do được ghi vào `abstain_reason`.
  - Xử lý lỗi: lỗi của tool được trả về cho model như một message và ghi vào trace. Lỗi provider ghi `status: error`, không thay bằng kết quả giả. Nội dung tài liệu được coi là dữ liệu: system prompt dặn model bỏ qua chỉ dẫn nằm trong passage.

Số liệu tổng hợp bằng [starter/agent_metrics.py](../../starter/agent_metrics.py) (qua `python -m src.metrics`) từ trace và nhãn thủ công ở `results/reviews/`. Có hai vòng chạy:

- **v1** ([results/agent_v1/](../../results/agent_v1/), nhãn ở [results/reviews/v1/](../../results/reviews/v1/)): bản đầu tiên, đã chạy cả dev và test.
- **v2** ([results/agent/](../../results/agent/)): v1 + **sửa lỗi định dạng citation ID** (chuẩn hóa `p239` → `p0239`, chỉ chấp nhận khi chunk thực sự có trong context). Vì lỗi này được phát hiện khi chấm test, test cũ coi như đã bị xem, nên mình viết **test2 mới** ([benchmark/test2.json](../../benchmark/test2.json): 8 câu có đáp án từ các trang chưa từng là gold + 3 câu thiếu evidence). v2 chạy lại dev rồi chạy test2 một lần.

| Cấu hình / split | Tool use | Số lượt gọi (search: số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| v1 A0_dense / dev | 25/25 | 1: 16, 2: 1, 3: 8 · 67 lượt LLM | 2 / 0 / 20 · 1.82 | 22/22 | 20/20 | 3/3 | 2/22 (q11, t01) |
| v1 A1_hybrid_neighbours / dev | 25/25 | 1: 20, 3: 5 · 59 lượt LLM | 0 / 2 / 20 · 1.91 | 25/27 | 22/22 | 3/3 | 0/22 |
| v1 A0_dense / test | 11/11 | 1: 7, 3: 4 · 30 lượt LLM | 1 / 0 / 7 · 1.75 | 7/7 | 7/7 | 3/3 | 1/8 (n02) |
| v1 A1_hybrid_neighbours / test | 11/11 | 1: 8, 3: 3 · 28 lượt LLM | 1 / 0 / 7 · 1.75 | 8/8 | 7/7 | 3/3 | 1/8 (n01) |
| **v2** A0_dense / dev | 25/25 | 1: 15, 2: 1, 3: 9 · 68 lượt LLM | 1 / 0 / 21 · 1.91 | 23/24 | 21/21 | 3/3 | 1/22 (t01) |
| **v2** A1_hybrid_neighbours / dev | 25/25 | 1: 21, 3: 4 · 58 lượt LLM | 0 / 2 / 20 · 1.91 | 23/25 | 22/22 | 3/3 | 0/22 |
| **v2** A0_dense / **test2** | 11/11 | 1: 5, 3: 6 · 34 lượt LLM | 1 / 1 / 6 · 1.62 | 7/8 | 8/8 | 3/3 | 0/8 |
| **v2** A1_hybrid_neighbours / **test2** | 11/11 | 1: 7, 3: 4 · 30 lượt LLM | 1 / 1 / 6 · 1.62 | 9/11 | 8/8 | 3/3 | 0/8 |

Retrieval-only trên test2 (câu hỏi gốc làm query): R0_dense Hit@1 5/8, Hit@3 7/8, Context Evidence 7/8; **R8_hybrid_neighbours 8/8, 8/8, 8/8**. Chấm context thủ công ([context_labels_test2.json](../../results/reviews/context_labels_test2.json)): R0 có 7 câu mức 2 và 1 câu mức 0 (m04: câu "identify and remove outliers before validating" không được lấy); R8 có 8 câu mức 2.

Context Evidence của agent (gom qua mọi lượt search): v1 A0 dev 20/22, A1 dev 22/22, A0 test 8/8, A1 test 8/8; v2 A0 dev 20/22, A1 dev 22/22, **A0 test2 6/8, A1 test2 7/8**. Chi phí và thời gian (gpt-4o-mini, $0.15 / $0.60 cho 1M token): v1 A0 dev $0.0090, 75.6 s (median 2.8 s/câu); v1 A1 dev $0.0101, 66.8 s (median 2.5 s); v1 test $0.0040 và $0.0049; v2 A0 dev $0.0093, 81.6 s; v2 A1 dev $0.0098, 71.3 s; test2 A0 $0.0044, 35.6 s; test2 A1 $0.0059, 34.0 s. Không có lỗi provider hay lỗi vector store nào.

Đọc từng câu trên test và test2 (mỗi bộ 8 câu có đáp án và 3 câu thiếu evidence, cả hai cấu hình); một số ví dụ:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| n04 (A1) | Có đáp án, diễn đạt lại | 1 search "noise-reduction strategy seven-day average…" → p0093:c01 hạng 1 | "…is called data smoothing… [p0093:c01]" | 2, citation đúng: chunk chứa "By calculating a seven-day average, we can smooth out these fluctuations" |
| n05 (A0 và A1) | Có đáp án, ô bảng | A0: 1 search "stopwords removal" → p0095:c01 vào context khi agent search | "filters out commonly occurring words that do not add meaning or context" + citation p0095:c01 | 2 ở cả hai cấu hình; ở bước retrieval-only, R0 có context mức 0 cho câu này. Agent viết query ngắn gọn hơn câu hỏi nên bù được lỗi retrieval |
| n07 (A1) | Có đáp án, điều kiện | 1 search → p0093:c04 hạng 1 và chunk kề p0094:c00 | "…against defined rules and standards… [p0093:c04], [p0094:c00]" | 2; cả 2 citation đều hỗ trợ (c00 nói về kiểm tra duplicate và giá trị thiếu) |
| **n01 (A1), lỗi** | Có đáp án | 1 search → p0239:c02 hạng 2 (chunk gold) | Model trả lời đúng nhưng cite `[B:openstax-intro-python:p239:c02]`, thiếu số 0 | 0: validator từ chối vì ID không có trong context → abstain |
| **n02 (A0), lỗi** | Có đáp án | 3 search, p0230:c01 trong context | Trả lời đúng, nhưng cite `p230:c01`, `p231:c01` | 0, cùng lỗi định dạng ID |
| u05 (A0 và A1) | Thiếu evidence | 3 search "Python version dictionaries preserve insertion order…" → không có passage liên quan | "Không đủ bằng chứng để trả lời." | Từ chối đúng, không thêm khẳng định nào |
| u06 (A0 và A1) | Thiếu evidence (sát chủ đề `copy()`) | 3 search "deep copy nested list copy module" | Từ chối | Đúng: corpus chỉ có `list.copy()` |
| **m02 (test2, A0 và A1), lỗi** | Có đáp án, ô bảng | Agent viết lại query thành "data cleaning erroneous values" / "identify outliers data cleaning" → không lấy được hàng *Validation* của Bảng 2.4 (p0090:c01) | "…handling of outliers during data cleaning" / "The data cleaning step identifies erroneous values…" | 0: sai, sách nói *data validation*. Retrieval-only với câu hỏi gốc thì lấy được p0090:c01 ở hạng 1 (cả R0 và R8) |
| m04 (test2, A1) | Có đáp án, điều kiện | 1 search "outliers before validating data" → p0094:c02 (gold) ở hạng 1 | "Outliers should be identified and handled properly before validating the data…" | 1: thiếu ý "remove" dù chunk được trích có câu đó |
| m07 (test2, A1) | Có đáp án | 1 search → p0240:c00 | "…my_dict = dict(my_list) [p0240:c00]" | 2 |
| v03 (test2) | Thiếu evidence | 3 search "sort dictionary by values…" | Từ chối | Đúng |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace agent: `results/agent/<config>-<split>/traces.json`. Mỗi câu lưu model, tham số, từng tool call (query, filter, kết quả, score, latency, chunk mới / chunk trùng / chunk bị bỏ), context cuối, câu trả lời, citation, token, thời gian và chi phí.
  - Metrics agent: `results/agent/<config>-<split>/summary.json` và `agent_metrics.json`.
  - Nhãn thủ công: `results/reviews/<config>-<split>.json` (điểm, đúng/sai cho từng citation, lý do).
  - Lệnh chạy: xem README.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Dev v1: A1 tốt hơn A0 (grounded 1.82 → 1.91, từ chối nhầm 2 → 0, ít lượt LLM hơn: 67 → 59), đổi lại chi phí cao hơn khoảng 12% do context dài hơn.
  - **Lỗi citation ID (v1 → v2).** gpt-4o-mini đôi khi bỏ số 0 đệm trong ID (`p0239` → `p239`), khiến validator từ chối câu trả lời đúng. Ở v1 lỗi này gây 3 lần từ chối nhầm: dev A0 q11, test A0 n02, test A1 n01. Sửa: `canon_id()` trong [src/agent.py](../../src/agent.py) chỉ chuẩn hóa số, mỗi lần chuẩn hóa ghi một event `citation_ids_canonicalized`, có unit test. Sau khi sửa, dev A0 q11 được chấp nhận (từ chối nhầm 2 → 1, grounded 1.82 → 1.91), và test2 không còn lỗi định dạng nào.
  - **test2 (đánh giá độc lập sau khi sửa):** hai cấu hình **hòa** (1.62). Về retrieval, R8 hơn R0 rõ (Hit@1 5/8 → 8/8, context 7/8 → 8/8), nhưng agent mất câu m02 ở cả hai cấu hình. Nguyên nhân: model **viết lại query** theo giả định sai ("data cleaning"), bỏ mất từ "validation" có trong câu hỏi gốc. Hướng sửa cho vòng sau (chưa làm vì test2 đã được xem): luôn search câu hỏi gốc ở lượt đầu, rồi mới cho model viết lại.
  - Trên test v1, hai cấu hình cũng hòa (1.75): lợi thế retrieval của R8 không chuyển thành câu trả lời tốt hơn, vì agent tự viết lại query nên bù được lỗi retrieval của dense (n05).
  - Lỗi khác: model có lúc trích sai chunk (A1 t03 trích một chunk chỉ có link video), tự thêm lý do không có trong sách (A1 t01), hoặc diễn đạt mất điều kiện (m04: "handled" thay vì "removed"). Một số câu trả lời có danh sách citation nhưng không ghi `[id]` ngay trong câu.
  - Giới hạn: benchmark nhỏ (mỗi test có 8 + 3 câu, nên mỗi câu bằng 12,5%), phần cơ bản chỉ có câu single-evidence. Câu multi-hop nằm ở phần nâng cao. Nhãn nháp được soạn với sự hỗ trợ của trợ lý AI bằng cách đọc đối chiếu từng citation với chunk; học viên chịu trách nhiệm kiểm tra lại. Chưa có người chấm thứ hai. Kết quả LLM có thể dao động giữa các lần chạy dù temperature = 0. Tách từ BM25 chỉ dùng regex đơn giản, không stemming.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện. Corpus 56 trang không đủ điều kiện 100 trang có bảng, ảnh và biểu đồ.
