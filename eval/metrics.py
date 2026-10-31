"""Deterministic retrieval metrics for annotated golden datasets."""
import math


def hit_rate_at_k(retrieved_ids, relevant_ids, k=5):
    relevant=set(relevant_ids or [])
    return float(any(x in relevant for x in (retrieved_ids or [])[:k])) if relevant else None


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant=set(relevant_ids or [])
    if not relevant: return None
    for rank, item in enumerate(retrieved_ids or [], 1):
        if item in relevant: return 1.0/rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    relevant=set(relevant_ids or [])
    if not relevant: return None
    dcg=sum((1.0/math.log2(rank+1)) for rank,item in enumerate((retrieved_ids or [])[:k],1) if item in relevant)
    ideal=sum(1.0/math.log2(rank+1) for rank in range(1,min(len(relevant),k)+1))
    return dcg/ideal if ideal else 0.0


def mean(values):
    vals=[v for v in values if isinstance(v,(int,float))]
    return sum(vals)/len(vals) if vals else None
