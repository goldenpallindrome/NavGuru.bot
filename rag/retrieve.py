from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from . import config
from .index import client,vectorize,ensure
from .lexical import search_bm25
from qdrant_client import models

_ranker=None
def reranker():
    global _ranker
    if _ranker is None: _ranker=Ranker(model_name=config.RERANK_MODEL,cache_dir="/home/rag/.cache/flashrank")
    return _ranker

def rrf(dense,lexical,limit):
    merged={}
    for arr in (dense,lexical):
        for rank,item in enumerate(arr,1):
            key=item["chunk_id"]
            if key not in merged: merged[key]=dict(item); merged[key]["rrf"]=0
            merged[key]["rrf"]+=1/(60+rank)
    return sorted(merged.values(),key=lambda x:x["rrf"],reverse=True)[:limit]

def build_filter(tenant_id, filters=None):
    filters=filters or {}; must=[models.FieldCondition(key="tenant_id",match=models.MatchValue(value=tenant_id))]
    def add_match(key, values):
        if values: must.append(models.FieldCondition(key=key,match=models.MatchAny(any=values)))
    add_match("document_id",filters.get("document_ids")); add_match("filename",filters.get("filenames"))
    add_match("extraction",filters.get("extraction_methods")); add_match("chunk_type",filters.get("chunk_types"))
    add_match("document_type",filters.get("document_types")); add_match("domain",filters.get("domains")); add_match("page",filters.get("pages"))
    for tag in filters.get("tags",[]): must.append(models.FieldCondition(key="tags",match=models.MatchValue(value=tag)))
    if filters.get("page_from") is not None or filters.get("page_to") is not None:
        must.append(models.FieldCondition(key="page",range=models.Range(gte=filters.get("page_from"),lte=filters.get("page_to"))))
    return models.Filter(must=must)

def search(question, tenant_id="default", filters=None):
    ensure(); qv=vectorize([question])[0]
    qfilter=build_filter(tenant_id,filters)
    hits=client().query_points(collection_name=config.COLLECTION,query=qv,
       query_filter=qfilter,limit=config.TOP_DENSE,with_payload=True,
       search_params={"hnsw_ef":96,"exact":False}).points
    dense=[dict(h.payload) for h in hits]
    lexical=search_bm25(question,config.TOP_BM25,tenant_id=tenant_id,filters=filters)
    candidates=rrf(dense,lexical,config.RERANK_TOP)
    if not candidates: return []
    passages=[{"id":c["chunk_id"],"text":c["text"]} for c in candidates]
    ranked=reranker().rerank(RerankRequest(query=question,passages=passages))
    byid={c["chunk_id"]:c for c in candidates}
    ordered=[]
    for r in ranked[:config.FINAL_K]:
        key=r.get("id")
        if key in byid: ordered.append(byid[key])
    return ordered or candidates[:config.FINAL_K]
