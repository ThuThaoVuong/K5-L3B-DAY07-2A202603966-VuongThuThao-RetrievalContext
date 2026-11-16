# Day 7 data

- `agent_cases.json`: 20 câu answerable chuyển từ benchmark Day 6 (gold bị loại khỏi agent input) và 4 câu unanswerable (2 dev, 2 test). Các `absence_reason` chỉ dùng để người làm kiểm tra/chấm, không truyền vào agent.
- `prepared/chunks.json`: được tạo bằng `starter/prepare_day7.py` từ bảng `rag_chunks` strategy C của Day 6. Không sửa IDs hoặc chunk text.
- `prepared/questions_answerable.json`: benchmark 20 câu có đáp án, có gold để evaluation; không truyền file này vào agent.

Corpus: Student Handbook for Global Engagement; 38 content pages theo Day 6 manifest.
