"""Dense, TF-IDF lexical and Reciprocal Rank Fusion hybrid retrieval."""
from __future__ import annotations
import math, re, unicodedata
from collections import Counter
from typing import Any
from day6_adapter import Day6DenseRetriever


def tokenize(text: str):
    return re.findall(r"[\w]+", unicodedata.normalize("NFC", text).casefold())

class TfidfRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        if not chunks: raise ValueError("chunks must not be empty")
        self.chunks=chunks
        counts=[Counter(tokenize(c["text"])) for c in chunks]
        df=Counter(term for row in counts for term in row)
        self.idf={term: math.log((1+len(chunks))/(1+n))+1 for term,n in df.items()}
        self.vectors=[self._vector(row) for row in counts]

    def _vector(self, counts):
        vec={t:n*self.idf[t] for t,n in counts.items() if t in self.idf}
        norm=math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {t:v/norm for t,v in vec.items()}

    def search(self, query: str, *, filters=None, top_k=3):
        q=self._vector(Counter(tokenize(query)))
        out=[]
        for c,v in zip(self.chunks,self.vectors):
            if filters and not all(c.get("metadata",{}).get(k)==val for k,val in filters.items()):
                continue
            score=sum(value*v.get(term,0.0) for term,value in q.items())
            if score>0:
                out.append({**c,"score":score,"method":"tfidf"})
        out.sort(key=lambda x:(-x["score"],x["id"]))
        return [{**c,"rank":i+1} for i,c in enumerate(out[:top_k])]

class HybridRetriever:
    def __init__(self, dense: Day6DenseRetriever, lexical: TfidfRetriever, rrf_k: int=60):
        self.dense=dense; self.lexical=lexical; self.rrf_k=rrf_k

    def search(self, query: str, *, filters=None, top_k=3):
        candidate_k=max(top_k*4,12)
        dense=self.dense.search(query,filters=filters,top_k=candidate_k)
        lexical=self.lexical.search(query,filters=filters,top_k=candidate_k)
        scores={}; by_id={}
        for rows in (dense,lexical):
            for rank,row in enumerate(rows,1):
                cid=row["id"]; scores[cid]=scores.get(cid,0.0)+1.0/(self.rrf_k+rank)
                by_id[cid]=row
        ranked=sorted(scores,key=lambda cid:(-scores[cid],cid))[:top_k]
        return [{**by_id[cid],"score":scores[cid],"rank":i+1,"method":"hybrid_rrf"}
                for i,cid in enumerate(ranked)]

class RetrievalRouter:
    def __init__(self, dense, lexical):
        self.dense=dense; self.lexical=lexical; self.hybrid=HybridRetriever(dense,lexical)
    def search(self, method, query, *, filters=None, top_k=3):
        if method=="dense": return self.dense.search(query,filters=filters,top_k=top_k)
        if method=="tfidf": return self.lexical.search(query,filters=filters,top_k=top_k)
        if method=="hybrid": return self.hybrid.search(query,filters=filters,top_k=top_k)
        raise ValueError(f"Unknown retrieval method: {method}")
