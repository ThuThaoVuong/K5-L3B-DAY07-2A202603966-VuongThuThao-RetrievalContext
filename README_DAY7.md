# Day 7 – Retrieval Context & Agentic RAG (tích hợp Day 6)

## Quyết định dữ liệu

Day 7 dùng đúng corpus **Student Handbook for Global Engagement** của Day 6. Manifest ghi nhận 46 trang PDF vật lý, trong đó 38 trang nội dung được chọn vào corpus. Mức 100 trang chỉ thuộc thử thách nâng cao trong `ADVANCED.md`; không phải điều kiện của phần cơ bản.

## Kế thừa Day 6

- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions).
- Vector store: PostgreSQL + pgvector.
- Chunking: strategy C mặc định (cùng boundaries với B; metadata filter ở retrieval time).
- Không rechunk, không re-embed và không thay chunk IDs: `prepare_day7.py` export trực tiếp records từ `rag_chunks`.
- Benchmark Day 6 giữ nguyên 20 câu, split 15 dev / 5 test. Day 7 bổ sung 2 câu không đủ bằng chứng cho mỗi split trong `data/agent_cases.json`.

## Setup (PowerShell)

1. Giải nén ZIP.
2. Tạo `.env` từ `.env.example`, điền API key và model agent nếu chạy agent.
3. Khởi động database từ `day6_reference/starter`:

   ```powershell
   cd day6_reference/starter
   docker compose up -d
   cd ../..
   ```

   Nếu bạn đang dùng database Day 6 hiện có, không cần tạo database mới; chỉ cần đảm bảo `DATABASE_URL` trỏ đúng database.
4. Cài dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements-day7.txt
   ```
5. Đảm bảo Day 6 đã ingest strategy C vào `rag_chunks`. Nếu chưa, chạy pipeline Day 6 trước (xem `day6_reference/README_DAY6.md`).
6. Export chunk và chuyển benchmark:

   ```powershell
   python starter/prepare_day7.py --strategy C
   ```

## Retrieval experiments

Chạy 3 phương pháp trên cùng corpus và dev set:

```powershell
python starter/retrieval_benchmark.py --split dev --method all --strategy C
```

Ba phương pháp: Dense (Day 6 pgvector), TF-IDF lexical, Hybrid RRF. Sau khi phân tích lỗi và chọn cấu hình bằng dev, mới chạy test:

```powershell
python starter/retrieval_benchmark.py --split test --method all --strategy C
```

Có thể thử ít nhất hai cấu hình cải thiện riêng: `--top-k 5 --budget 2400` và `--top-k 8 --budget 3200`, ghi giả thuyết và kết quả; không mặc định rằng tăng top-k luôn tốt hơn.

## Agent thật

Điền `OPENROUTER_API_KEY` (hoặc `OPENAI_API_KEY`) và `AGENT_MODEL` trong `.env`, sau đó:

```powershell
python starter/agent_runner.py --split dev --method hybrid
```

Sau khi chốt bằng dev:

```powershell
python starter/agent_runner.py --split test --method hybrid
```

Agent chỉ nhận câu hỏi và kết quả retrieval. `expected_answer`, `gold_evidence`, `absence_reason` không được truyền vào model. Đọc thủ công citation và điền review labels theo `BENCHMARK.md`.

## Output

- `data/prepared/chunks.json`: export trực tiếp từ Day 6 DB.
- `data/prepared/questions_answerable.json`: chuyển benchmark Day 6 sang contract Day 7.
- `results/retrieval/<split>/<method>/`: traces và summary.
- `results/agent/<split>/traces.json`: tool calls, context, answer, citation, lỗi.

Không có benchmark score giả được tạo sẵn. Chỉ số sẽ được sinh khi chạy trên database/model thực tế.
