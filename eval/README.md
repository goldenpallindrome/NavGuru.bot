# RAG evaluation pipeline

This is a separate offline evaluation workflow. It does not add evaluation work to the live `/chat` path.

## 1. Golden dataset

Copy `datasets/golden.template.jsonl` to `datasets/golden.v1.jsonl`. Each JSONL row is one expert-labelled test case:

- `id`: stable case identifier.
- `category` (optional): test slice such as `single_hop_lookup`, `multi_hop`, `table_qa`, `ocr`, `follow_up`, `metadata_filter`, `domain_mismatch`, or `no_answer`; reports include per-category metrics.
- `question`, `tenant_id`, `filters`: the exact request and access scope.
- `expected_domain`, `expected_route`: expected query-routing behavior (`answer`, `domain_mismatch`, `clarify`).
- `ground_truth`: expert-written ideal answer; required for RAGAS answer metrics.
- `reference_contexts`: expert-selected supporting passages; required for context recall/precision.
- `relevant_chunk_ids`: IDs of chunks judged relevant; required for deterministic Hit@K, MRR and nDCG.

Replace all `REPLACE_WITH_REAL_CHUNK_ID` values and illustrative reference text. A template is not a valid benchmark until an SME/domain expert reviews it against the source PDFs. Keep sensitive data out of test sets or use masked/synthetic examples. Version datasets (v1, v2) and retain a change log.

A useful initial benchmark is 50–100 questions per domain, with a balanced mix of simple lookups, multi-hop questions, tables, OCR pages, ambiguous follow-ups, no-answer questions, cross-domain questions and metadata-filtered questions. Include negative and adversarial cases. Have a second reviewer adjudicate high-impact labels.

## 2. Run deterministic routing + retrieval evaluation

Run from the project root after Docker services are up, documents are indexed and `.env` is configured:

```bash
python -m eval.run_eval --dataset eval/datasets/golden.v1.jsonl --mode retrieval
```

This executes the real query-domain gate and filtered hybrid retrieval. It reports domain/route accuracy, Hit@5, MRR and nDCG@5 when relevant chunk IDs are labelled. Missing relevance labels are reported as `null`, not misleading zeros. Output JSON and CSV files are written under `eval/results/` (ignored by git by default).

## 3. Run full answer evaluation with RAGAS

Install optional dependencies separately from production:

```bash
pip install -r requirements-eval.txt
python -m eval.run_eval --dataset eval/datasets/golden.v1.jsonl --mode full --ragas
```

`--mode full` runs the project's actual answer-generation flow. `--ragas` then computes:

- **Faithfulness**: whether answer claims are supported by retrieved context.
- **Answer relevancy**: whether the answer addresses the question.
- **Context precision**: whether retrieved context is relevant relative to the reference.
- **Context recall**: whether reference information is present in retrieved context.
- **Answer correctness**: semantic agreement with the expert reference answer.

RAGAS uses the configured OpenAI-compatible LLM endpoint by default (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). Optional overrides: `RAGAS_API_KEY`, `RAGAS_BASE_URL`, `RAGAS_LLM_MODEL`, `RAGAS_EMBEDDING_MODEL`, `RAGAS_EMBEDDING_BASE_URL`. If your provider does not expose an OpenAI-compatible embeddings endpoint, configure a compatible embedding provider or adapt the evaluator. RAGAS is a judge-based metric suite: scores are estimates, not ground truth, and can be affected by judge model quality, prompt, language and nondeterminism. Keep judge model/version fixed when comparing runs.

## 4. Recommended metric interpretation

| Area | Metric | What it diagnoses |
|---|---|---|
| Routing | Domain accuracy / route accuracy | Wrong-domain acceptance or valid-query rejection |
| Retrieval | Hit@5 | At least one labelled relevant chunk found in top 5 |
| Retrieval | MRR | Rank of the first relevant chunk |
| Retrieval | nDCG@5 | Ranking quality with binary relevance labels |
| Generation | Faithfulness | Unsupported statements / hallucination risk |
| Generation | Answer relevancy | Off-topic or incomplete answers |
| Retrieval | Context precision | Noise in retrieved passages |
| Retrieval | Context recall | Missing required evidence |
| Generation | Answer correctness | Agreement with expert reference |
| Integrity | Citation integrity rate | Existing deterministic filename/page validation only; not entailment proof |

Do not use one aggregate score as the only release gate. Set thresholds based on a pilot and risk level. As a starting point, investigate any domain-routing false acceptance, any high-severity unsupported claim, and regressions greater than 3–5 percentage points on key metrics. For regulated/high-impact use, require expert review and stricter thresholds.

## 5. Regression and experiment protocol

1. Freeze a golden dataset version and source corpus snapshot.
2. Record code commit, corpus/index version, embedding model, chunk size/overlap, dense/BM25 candidate counts, reranker, final K, generation model and RAGAS judge model.
3. Run baseline and candidate configurations on the same cases.
4. Compare per-case failures, not only averages; stratify by domain, OCR/native, tables, question type and metadata filter.
5. Add every production failure (after PII scrubbing and review) as a new golden case.
6. Re-run after changes to extraction, chunking, embeddings, retrieval, prompts, classifier or model.

## Limitations

The included template is illustrative and does not contain the user's actual source facts or real relevant chunk IDs. Meaningful RAGAS and retrieval scores require expert-curated answers, reference contexts and chunk relevance labels. Full evaluation invokes the configured LLM and may incur provider cost/latency. Evaluation is offline and must run against a test/staging corpus, not during user requests. No production quality threshold is asserted by this starter.

## 6. Run in the provided Docker environment

The optional `evaluation` build target installs RAGAS dependencies without adding them to the production API image. First create your reviewed dataset file, then run:

```bash
# Deterministic routing + retrieval metrics
mkdir -p eval/results
docker compose --profile evaluation run --rm eval \
  python -m eval.run_eval --dataset eval/datasets/golden.v1.jsonl --mode retrieval

# Full answer + RAGAS metrics (uses configured LLM endpoint and may incur cost)
docker compose --profile evaluation run --rm eval \
  python -m eval.run_eval --dataset eval/datasets/golden.v1.jsonl --mode full --ragas
```

The evaluation container shares the local `data/` index files and model cache, and writes reports to `eval/results/`. It is a one-shot offline job; it does not start a public API endpoint.
