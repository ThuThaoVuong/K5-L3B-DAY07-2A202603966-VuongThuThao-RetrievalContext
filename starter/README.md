# Starter Day 7 – tích hợp Day 6

Các module chính:
- `prepare_day7.py`: export chunks nguyên bản từ Day 6 pgvector và chuyển benchmark.
- `day6_adapter.py`: dùng lại embedder và PgVectorStore Day 6.
- `retrieval_methods.py`: Dense, TF-IDF, Hybrid RRF.
- `retrieval_benchmark.py`: so sánh 3 phương pháp, xuất Hit@1, Evidence Hit@3, Context Evidence Rate và traces.
- `agentic.py`: vòng lặp agent starter theo hợp đồng Day 7.
- `agent_runner.py`: controller dùng model OpenAI-compatible thật; ghi tool calls/context/citation/errors.
- `baseline.py`, `agent_metrics.py`: baseline TF-IDF và tổng hợp nhãn audit thủ công.

Chạy theo `README_DAY7.md`. Không có mock vector hoặc kết quả benchmark dựng sẵn.
