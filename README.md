# Day 7 — Agentic RAG

> **Bài làm của học viên** — Dương Thị Huyền (2A202604077). Báo cáo: [report/individual/2A202604077_DuongThiHuyen.md](report/individual/2A202604077_DuongThiHuyen.md). Đề bài gốc giữ nguyên bên dưới đường kẻ.

## Bài làm: cách chạy lại

Yêu cầu: Python 3.12, Docker Desktop, repo Day 6 nằm cạnh repo này (`../K5-L3-Day06-DuongThiHuyen-2A202604077-DataIngestion`, đã ingest index A/B vào pgvector), OpenAI API key. Chi phí toàn bộ thí nghiệm ≈ $0.03.

```powershell
# 1. Vector store Day 6 (PostgreSQL 16 + pgvector 0.8.7, bge-small-en-v1.5, 384 chiều)
cd ..\K5-L3-Day06-DuongThiHuyen-2A202604077-DataIngestion\starter; docker compose -p day6_student_starter up -d --wait; cd -
# 2. Môi trường (phiên bản chính xác: requirements.lock.txt)
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env        # điền OPENAI_API_KEY; không commit .env
$env:PYTHONUTF8=1                  # Windows: test starter đọc file không chỉ định encoding
# 3. Dữ liệu + test
.venv\Scripts\python -m src.prepare_data                       # xuất 189 chunk từ pgvector, build benchmark, kiểm tra gold
.venv\Scripts\python -m unittest discover -s tests -v          # 22 test offline (mock chỉ kiểm tra code)
# 4. Retrieval benchmark (không gọi LLM)
.venv\Scripts\python -m src.eval_retrieval --split dev                                       # R0..R8
.venv\Scripts\python -m src.eval_retrieval --split test --configs R0_dense R8_hybrid_neighbours
# 5. Agent thật (gpt-4o-mini)
.venv\Scripts\python -m src.run_agent --split dev  --config A0_dense
.venv\Scripts\python -m src.run_agent --split dev  --config A1_hybrid_neighbours
.venv\Scripts\python -m src.run_agent --split test --config A0_dense
.venv\Scripts\python -m src.run_agent --split test --config A1_hybrid_neighbours
# 6. Tổng hợp nhãn thủ công
.venv\Scripts\python starter/agent_metrics.py --cases data/cases.json --traces results/agent/A1_hybrid_neighbours-test/traces.json --reviews results/reviews/A1_hybrid_neighbours-test.json --split test --output results/agent/A1_hybrid_neighbours-test/agent_metrics.json
```

| Thành phần | Vị trí |
|---|---|
| Code: config, dữ liệu/hit, retriever (dense, BM25, RRF, rerank, chunk kề), context, agent | [src/](src/) |
| Corpus đã xuất + benchmark | [data/chunks.json](data/chunks.json), [data/questions.json](data/questions.json), [data/cases.json](data/cases.json), [benchmark/](benchmark/) |
| Trace và summary retrieval | `results/retrieval/<config>-<split>/` |
| Trace và metrics agent | `results/agent/<config>-<split>/` |
| Nhãn chấm thủ công (câu trả lời, citation, abstention, context) | [results/reviews/](results/reviews/) |

Corpus: OpenStax, CC BY-NC-SA 4.0 (Rice University); văn bản trích từ các trang đã chọn, xem manifest của Day 6.

Phần cơ bản có hai vòng: **v1** ([results/agent_v1/](results/agent_v1/)) và **v2** ([results/agent/](results/agent/)). v2 là v1 + sửa lỗi định dạng citation ID, đánh giá trên **test2** mới ([benchmark/test2.json](benchmark/test2.json)): `python -m src.run_agent --split test2 --config A1_hybrid_neighbours`, `python -m src.metrics --run A1_hybrid_neighbours-test2`.

### Thử thách nâng cao (tài liệu 569 trang, đa phương thức)

Tài liệu: *Principles of Data Science* (OpenStax, 2024 web edition, CC BY-NC-SA 4.0), 569 trang PDF, SHA-256 `6b47205459a2f4e25a2b2a86369e674a26a5ae566a4cc428dd31047670692398`. Tải từ https://assets.openstax.org/oscms-prodcms/media/documents/Principles-of-Data-Science-WEB.pdf (script `scripts/download_corpus.py` của Day 6 tải đúng file này và kiểm tra hash). File PDF không được commit.

```powershell
.venv\Scripts\python -m advanced.extract          # text layer + bảng (find_tables) + vùng hình + caption + render trang -> advanced/data/pages.jsonl
.venv\Scripts\python -m advanced.describe         # gpt-4.1-mini vision mô tả bảng/biểu đồ/sơ đồ từng trang -> advanced/data/visual/ (~$1)
.venv\Scripts\python -m advanced.index            # chunk text/table/visual -> bge-small 384-d -> pgvector bảng adv_chunks
.venv\Scripts\python -m advanced.eval_retrieval --split dev                         # so sánh retrieval (không dùng LLM)
.venv\Scripts\python -m advanced.run --split dev   --config V4_mm_hybrid_mustview    # round 1 (V0..V4)
.venv\Scripts\python -m advanced.run --split test  --config V4_mm_hybrid_mustview    # test round 1: 25/30
.venv\Scripts\python -m advanced.run --split dev2  --config V6_round2_zoom_gpt5mini  # dev2 = test round 1
.venv\Scripts\python -m advanced.run --split test2 --config V6_round2_zoom_gpt5mini  # test2 mới: 29/30 = 96.7%
```

- Benchmark: [advanced/data/benchmark.json](advanced/data/benchmark.json) (dev 17 + 4, test round 1: 24 + 6) và [advanced/data/benchmark_round2_test.json](advanced/data/benchmark_round2_test.json) (test2: 24 + 6). Các file này được commit trước khi chạy.
- Lý do chọn cấu hình: [advanced/data/SELECTION.md](advanced/data/SELECTION.md).
- Trace: `results/advanced/<config>-<split>/`. Nhãn đạt/không đạt kèm lý do: `results/advanced/reviews/`.
- Ảnh render và crop (`advanced/cache/`) không được commit; tạo lại bằng `advanced.extract`.

---

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
