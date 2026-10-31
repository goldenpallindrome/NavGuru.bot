from eval.schema import load_golden


def test_template_dataset_loads():
    rows=load_golden("eval/datasets/golden.template.jsonl")
    assert len(rows)>=4
    assert any(x["expected_route"]=="domain_mismatch" for x in rows)
