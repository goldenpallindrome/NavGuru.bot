from eval.metrics import hit_rate_at_k, reciprocal_rank, ndcg_at_k


def test_retrieval_metrics():
    ranked=["a","b","c"]
    relevant=["b","c"]
    assert hit_rate_at_k(ranked,relevant,2)==1.0
    assert reciprocal_rank(ranked,relevant)==0.5
    assert 0 < ndcg_at_k(ranked,relevant,3) <= 1


def test_metrics_without_labels_are_not_zero():
    assert hit_rate_at_k(["a"],[],5) is None
    assert reciprocal_rank(["a"],[]) is None
    assert ndcg_at_k(["a"],[],5) is None
