# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
- Vector store, embedding model, phiên bản và số chiều:
- Agent model, cách chạy và phân chia dev/test:
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:

- Nguồn dữ liệu: Student Handbook for Global Engagement (PDF), kế thừa từ Day 6.
- Quyền sử dụng: Sử dụng tài liệu phục vụ mục đích học tập và nghiên cứu.
- Số trang: 38 trang nội dung được chọn.
- Số chunk: 302 chunks, sử dụng Chunking Strategy C.
- Vector store: PostgreSQL 16 + pgvector.
- Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
- Embedding dimensions: 384.
- Vector strategy: C.
- Agent model: openai.
- Phân chia benchmark: 20 câu hỏi từ benchmark Day 6, gồm 15 câu dev và 5 câu test.
- Gold evidence: Được lưu riêng trong benchmark/evaluation, không truyền trực tiếp vào Agent.
- Cách xác định evidence: Dựa trên source_id, số trang (page) và đoạn nội dung bằng chứng (evidence) được xác định trước trong benchmark.
- Tiêu chí đánh giá:
  - Hit@1: Kết quả đứng đầu có chứa gold evidence hay không.
  - Evidence Hit@3: Có ít nhất một kết quả trong top 3 chứa gold evidence hay không. 
  - Context Evidence Rate: Context cuối cùng được cung cấp có chứa gold evidence hay không.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense, top-k=3, context=1800 | Retrieval bằng semantic similarity | dev | 0/15 (0%) | 0/15 (0%) | 0/15 (0%) | 126.52 ms / $0 |
| Phương pháp 2 | TF-IDF, top-k=3, context=1800 | Thử lexical retrieval dựa trên từ khóa | dev | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 1.88 ms / $0 |
| Phương pháp 3 | Hybrid RRF (Dense + TF-IDF), top-k=3, context=1800 | Kết hợp Dense và Lexical bằng Reciprocal Rank Fusion | dev | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 120.35 ms / $0 |
| Cải thiện 1 | | | dev | | | | |
| Cải thiện 2 | | | dev | | | | |
| Cấu hình đã chọn | | | test | | | | |
| Baseline | | | test | | | | |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
