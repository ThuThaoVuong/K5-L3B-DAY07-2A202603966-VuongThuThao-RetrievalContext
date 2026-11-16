"""Run comparable retrieval experiments over Day 6's same corpus/questions."""
from __future__ import annotations
import argparse,json,os,time,statistics
from pathlib import Path
from dotenv import load_dotenv
from day6_adapter import Day6DenseRetriever
from retrieval_methods import TfidfRetriever,RetrievalRouter
from baseline import hit,question_hit,pack_context,validate

ROOT=Path(__file__).resolve().parents[1]

def main():
    load_dotenv(ROOT/".env")
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["dev","test"],required=True)
    parser.add_argument("--method",choices=["dense","tfidf","hybrid","all"],default="all")
    parser.add_argument("--strategy",choices=["A","B","C"],default=os.getenv("DAY6_STRATEGY","C"))
    parser.add_argument("--top-k",type=int,default=3)
    parser.add_argument("--budget",type=int,default=1800)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data"/"prepared")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results"/"retrieval")
    args=parser.parse_args()
    db=os.getenv("DATABASE_URL")
    if not db: raise SystemExit("DATABASE_URL is missing in .env")
    chunks=json.loads((args.data_dir/"chunks.json").read_text(encoding="utf-8"))
    all_q=json.loads((args.data_dir/"questions_answerable.json").read_text(encoding="utf-8"))
    questions=[q for q in all_q if q["split"]==args.split]
    if not questions: raise SystemExit(f"No questions for split={args.split}")
    validate(chunks,questions)
    dense=Day6DenseRetriever(db,args.strategy,os.getenv("EMBEDDING_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    lexical=TfidfRetriever(chunks)
    router=RetrievalRouter(dense,lexical)
    methods=["dense","tfidf","hybrid"] if args.method=="all" else [args.method]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for method in methods:
        traces=[]
        for q in questions:
            started=time.perf_counter()
            results=router.search(method,q["question"],filters=q.get("filter"),top_k=max(3,args.top_k))
            elapsed=(time.perf_counter()-started)*1000
            marked=[{**r,"hit":hit(r,q["gold"])} for r in results]
            # Standardize comparison at top-3 even if a larger retrieval top-k is configured.
            top3=marked[:3]
            context,context_ids=pack_context(marked,budget=args.budget)
            context_rows=[r for r in marked if r["id"] in context_ids]
            traces.append({"query_id":q["id"],"split":q["split"],"question":q["question"],
                "method":method,"strategy":args.strategy,"embedding_model":dense.model_name,
                "embedding_dimensions":dense.dimensions,"top_k":args.top_k,"filter":q.get("filter"),
                "gold":q["gold"],"results":marked,"context":context,"context_ids":context_ids,
                "hit1":question_hit(marked[:1],q),"evidence_hit3":question_hit(top3,q),
                "context_evidence":question_hit(context_rows,q),
                "context_chars":len(context),"elapsed_ms":elapsed})
        n=len(traces)
        summary={"method":method,"split":args.split,"n":n,
            "hit1":{"n":sum(t["hit1"] for t in traces),"N":n,"rate":sum(t["hit1"] for t in traces)/n},
            "evidence_hit3":{"n":sum(t["evidence_hit3"] for t in traces),"N":n,"rate":sum(t["evidence_hit3"] for t in traces)/n},
            "context_evidence_rate":{"n":sum(t["context_evidence"] for t in traces),"N":n,"rate":sum(t["context_evidence"] for t in traces)/n},
            "median_ms":statistics.median(t["elapsed_ms"] for t in traces),
            "avg_context_chars":statistics.mean(t["context_chars"] for t in traces),
            "embedding_model":dense.model_name,"embedding_dimensions":dense.dimensions,
            "vector_strategy":args.strategy,"top_k":args.top_k,"context_budget_chars":args.budget,
            "service_cost_usd":0 if method in ("dense","tfidf","hybrid") else None}
        out=args.output_dir/args.split/method; out.mkdir(parents=True,exist_ok=True)
        (out/"traces.json").write_text(json.dumps(traces,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
