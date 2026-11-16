"""Export existing Day 6 Strategy C chunks and convert its frozen benchmark.

No re-chunking or re-embedding occurs here. Chunk IDs/text/page/source/metadata
are read directly from the Day 6 pgvector index.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = Path(os.getenv("DAY6_PROJECT_DIR", str(ROOT / "day6_reference"))).resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_chunks(database_url: str, strategy: str, output: Path):
    sql = """SELECT chunk_id, document_id, source, page, text, metadata
             FROM rag_chunks WHERE strategy = %s ORDER BY document_id, page, chunk_id"""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, (strategy,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"Không tìm thấy chunks strategy={strategy!r} trong rag_chunks. Hãy chạy Day 6 ingestion trước.")
    chunks=[]
    for chunk_id, document_id, source, page, text, metadata in rows:
        chunks.append({"id": chunk_id, "source_id": source, "source": source,
                       "page": int(page), "text": text, "metadata": {**(metadata or {}), "document_id": document_id}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Exported {len(chunks)} Day 6 chunks from strategy {strategy} -> {output}")
    return chunks


def convert_questions(benchmark_path: Path, output: Path):
    raw=load_json(benchmark_path)
    converted=[]
    for q in raw["questions"]:
        gold=[{"source_id": e["source"], "page": int(e["page"]), "evidence": e["text"]}
              for e in q.get("gold_evidence", [])]
        # Day 6 stored some long page-level quotes. For Day 7's chunk-level
        # evaluation, use concise verbatim spans from the original quote that
        # remain inside Strategy C chunks. Multi-part questions require every
        # span in a group, not merely one matching chunk.
        required_spans = {
            "Q008": [
                "Every research study requires IRB approval prior to commencement.",
                "The conditions for ethical approval are more stringent for research studies in vulnerable populations such as children, incarcerated persons, pregnant women, and persons with impaired decisional capacity as well as conditions that are be stigmatizing such as HIV.",
            ],
            "Q007": [
                "Fiscal sustainability of operations. At the start, the leader of a research or service project may be particularly concerned with fiscal sustainability: Are there enough resources and funding to keep the project going?",
                '"Putting oneself out of business." This more advanced conceptualization of sustainability means implementing a research or service project so effectively that it no longer needs your input or support or can be sustained by local people without outside assistance.',
            ],
            "Q017": [
                "All parties (including domestic advisors and international collaborators) must be a part of the discussion and exploitive behavior must be avoided.",
                "The influence of local knowledge and local products should guide the decision in determining the ownership of intellectual property.",
            ],
            "Q019": [
                "Because there are different interpretations, always strive to maintain and display the highest standard possible when considering confidentiality abroad.",
                "Refrain from posting and publishing confidential or identifying images, information, and quotations in public domains, including digital and/or social media (i.e. Facebook, MySpace, Twitter, blogs, etc.), without appropriate informed consent.",
            ],
        }
        if q["question_id"] in required_spans:
            source = q["gold_evidence"][0]["source"]
            page = int(q["gold_evidence"][0]["page"])
            gold = [{"source_id": source, "page": page, "evidence": span}
                    for span in required_spans[q["question_id"]]]
        item={"id": q["question_id"], "split": q["split"], "question": q["question"],
              "expected_answer": q["expected_answer"], "gold": gold, "gold_groups": [gold]}
        filt=q.get("retrieval_filter")
        if filt:
            item["filter"]={filt["field"]: filt["value"]}
        converted.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Converted {len(converted)} Day 6 benchmark questions -> {output}")
    return converted


def main():
    load_dotenv(ROOT / ".env")
    load_dotenv(DAY6_ROOT / "starter" / ".env", override=False)
    parser=argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["A","B","C"], default=os.getenv("DAY6_STRATEGY","C"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data"/"prepared")
    args=parser.parse_args()
    if not args.database_url: raise SystemExit("Thiếu DATABASE_URL. Điền trong .env hoặc Day6 starter/.env")
    export_chunks(args.database_url, args.strategy, args.output_dir/"chunks.json")
    convert_questions(DAY6_ROOT/"benchmark"/"questions.json", args.output_dir/"questions_answerable.json")
    print("Gold chỉ nằm trong benchmark/evaluation; không được truyền vào agent.")

if __name__=="__main__": main()
