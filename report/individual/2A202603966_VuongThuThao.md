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
- Phân chia benchmark: 24 câu hỏi với 15 câu dev có đáp án, 2 câu dev không đáp án, 5 câu test có đáp án, 2 câu test không đáp án.
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
| Cải thiện 1 | TF-IDF, top-k=5, context=2400 | Tăng số lượng chunks truy xuất từ 3 lên 5 và mở rộng context budget từ 1800 lên 2400 ký tự | dev | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 1.99 ms / $0 |
| Cải thiện 2 | TF-IDF top-20 + Cross-Encoder Reranking, final top-k=5, context=2400 | Truy xuất 20 candidate chunks bằng TF-IDF, sau đó sử dụng Cross-Encoder để đánh giá và sắp xếp lại mức độ liên quan, giữ lại 5 chunks tốt nhất | dev | 8/15 (53.33%) | 10/15 (66.67%) | 11/15 (73.33%) | 612.50 ms / $0 |
| Cấu hình đã chọn | TF-IDF top-20 + Cross-Encoder Reranking, final top-k=5, context budget=2400 | Truy xuất 20 candidate chunks bằng TF-IDF, sau đó sử dụng Cross-Encoder để đánh giá và sắp xếp lại mức độ liên quan, giữ lại 5 chunks tốt nhất | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 704.04 ms / $0 |
| Baseline | Dense, top-k=3, context=1800 | Retrieval bằng Dense semantic similarity | test | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 128.21 ms / $0 |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

- Một cấu hình có thể vừa là phương pháp mới, vừa là cải thiện vì nó có thể thay đổi phương pháp retrieval so với baseline, đồng thời bổ sung kỹ thuật để nâng cao chất lượng truy xuất.

- So sánh kết quả trên tập test

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|
| Baseline – Dense | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 128.21 ms / $0 |
| Cấu hình đã chọn – TF-IDF + Cross-Encoder | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 704.04 ms / $0 |


- Trên dev, cấu hình đã chọn đạt Hit@1: 8/15 (53.33%), Evidence Hit@3: 10/15 (66.67%), Context Evidence Rate: 11/15 (73.33%).
- Chọn TF-IDF top-20 kết hợp Cross-Encoder reranking vì đạt chất lượng evidence tốt nhất trong các cấu hình đã thử.
- Đổi lại, thời gian xử lý tăng do Cross-Encoder mất khoảng 703.60 ms. Chi phí dịch vụ là $0.
- Trên test, cấu hình đạt Evidence Hit@3 và Context Evidence Rate 5/5 (100%), tốt hơn baseline Dense.


## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: Sử dụng tool `Search`, mỗi lần lấy Top-3 chunks làm context. Trace ghi nhận 1–3 lượt gọi/query. Nếu vẫn không đủ bằng chứng sau khi thử lại, Agent trả về: "Không đủ bằng chứng để trả lời".

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---:|---|---:|---:|---:|---:|
| Hybrid / Dev | Search | 25 | 0: 1, 1: 4, 2: 4; TB: 1,33 | 9/9 | 9/9 | 2/8 | 6/8 |
| Hybrid / Test | Search | 9 | 0: 0, 1: 1, 2: 3; TB: 1,75  | 4/4 | 4/4 | 2/3 | 1/3 |
| Tổng cộng | Search | 34 | 0: 1, 1: 5, 2: 7; TB: 1,46 | 13/13 | 13/13 | 4/11 | 7/11 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| Q004 | Có thể trả lời | 3 calls, 9 kết quả | Abstain, không citation | Từ chối nhầm; context chưa được khai thác hiệu quả |
| Q005 | Có thể trả lời | 1 call, 3 kết quả | Abstain, không citation | Từ chối nhầm; context có thông tin liên quan |
| Q019 | Có thể trả lời | 1 call, 3 kết quả | Abstain, không citation | Từ chối nhầm; context có nội dung về bảo mật và đồng thuận |
| NA_DEV_01 | Không thể trả lời | 3 calls, 9 kết quả | Abstain | Abstention đúng; không đủ bằng chứng |
| NA_DEV_02 | Không thể trả lời | 3 calls, 9 kết quả | Abstain | Abstention đúng; không đủ bằng chứng |
| NA_TEST_01 | Không thể trả lời | 1 call, 3 kết quả | Abstain | Abstention đúng; không đủ bằng chứng |
| NA_TEST_02 | Không thể trả lời | 3 calls, 9 kết quả | Abstain | Abstention đúng; không đủ bằng chứng |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Trace Dev: `results/agent/dev/traces.json`.
  - Trace Test: `results/agent/test/traces.json`.
  - Model: `gpt-4o-mini`; method: `hybrid`.
  - Tổng số token sử dụng: 20.722 tokens (OpenAI).
  - Thời gian trung bình Dev: 4,07 giây; Test: 4,91 giây.
  - Chi phí API phụ thuộc vào đơn giá token của model tại thời điểm sử dụng; chưa có dữ liệu chi tiết token đầu vào và đầu ra để tính chính xác.

- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - Tổng cộng trả lời được 13/24 câu, abstain 11/24 câu.
  - 4/4 câu không thể trả lời được abstain đúng; 7 câu có thể trả lời bị từ chối nhầm.
  - Citation của 13/13 câu trả lời đều trỏ đến kết quả retrieval, nhưng chưa đánh giá mức độ citation thực sự hỗ trợ nội dung.
  - Chưa có trace baseline nên chưa thể so sánh trước/sau. Grounded answer đã được chấm thủ công, đạt trung bình 1,46/2.

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
