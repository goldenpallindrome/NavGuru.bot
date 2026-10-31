# Private PDF Agentic RAG — CPU + Redis + hybrid inference

## Metadata-aware retrieval and tenant scoping

Every indexed chunk carries structured metadata: `tenant_id`, `document_id`, `filename`, `page`, `domain`, `extraction` (`native` or `page_ocr`), `chunk_type`, `document_type`, `tags`, `source`, and `created_at` where supplied. Qdrant payload indexes are created for the frequently filtered fields. The SQLite FTS5/BM25 path applies the same tenant and metadata constraints, so hybrid fusion does not re-introduce chunks excluded by the vector filter.

Both `/chat` and `/sessions/{session_id}/chat` accept optional metadata filters. Example session request body:

```json
{
  "question": "What does the policy say about retention?",
  "filters": {
    "document_types": ["policy"],
    "tags": ["retention"],
    "filenames": ["data_policy.pdf"],
    "page_from": 1,
    "page_to": 40,
    "extraction_methods": ["native", "page_ocr"]
  }
}
```

Supported filters: `document_ids`, `filenames`, `pages`, `page_from`, `page_to`, `extraction_methods`, `chunk_types`, `document_types`, `domains`, and `tags`. Filters are allow-listed and bounded. The tenant constraint is always applied from the JWT `tenant_id` claim (falling back to `sub`); the service API key uses the `default` tenant. Ingestion is an admin-only operation and accepts `tenant_id`, `document_type`, `tags`, and `source` metadata as query parameters. Re-ingest documents after upgrading so the FTS metadata schema and Qdrant payloads are populated.

**Tenant isolation is not optional:** issue JWTs with a stable tenant claim, ingest each tenant's files under that same tenant ID, and never let end users choose or override their tenant. The admin ingestion credential is trusted to assign tenant ownership. For user-level sharing within a tenant, add a document ACL/access-group filter to both Qdrant and FTS before enabling mixed-visibility documents.

## Query understanding, intent routing and knowledge-base validation (added)

Every request now passes through a lightweight deterministic query-understanding stage before document retrieval:

- Detects a broad domain (medical, finance, legal, HR, technology, insurance, education, or unknown).
- Classifies common intent types (fact lookup, explanation, summary, comparison, calculation, procedural).
- Returns matched terms and a confidence signal for observability.
- Checks domain availability using tenant-scoped indexed chunk counts.
- Stops early with a clear domain-mismatch response when a confident query domain is absent from a labelled corpus; ambiguous multi-domain queries can ask for clarification.
- Automatically applies the detected domain as a retrieval filter when that domain exists in the tenant corpus. The caller can also specify `filters.domains` explicitly.

### Assign a domain during ingestion

Domain labels are essential for reliable mismatch detection. Set `domain` when queuing each PDF. Supported values: `medical`, `finance`, `legal`, `hr`, `technology`, `insurance`, `education`, `general`.

Example ingestion request (admin credential):

```bash
curl -X POST http://127.0.0.1:8000/admin/ingest/annual_report.pdf \
 -H 'Authorization: Bearer YOUR_SERVICE_API_KEY' \
 -H 'Content-Type: application/json' \
 -d '{"tenant_id":"tenant-a","domain":"finance","document_type":"annual_report","tags":["annual","accounts"]}'
```

The `/admin/ingest/{filename}` endpoint accepts metadata as a JSON body. Existing documents need re-ingestion to populate the new `domain` field in Qdrant and FTS5. Documents ingested as `general` are treated as unclassified and will not trigger a confident domain-mismatch rejection.

### Example mismatch behavior

If a tenant has only documents labelled `finance` and the user asks "What are the symptoms of diabetes?", the classifier detects `medical`, checks tenant-scoped domain counts, and returns a domain-mismatch response without vector search, BM25, reranking or LLM generation. It does not invent a medical answer from financial evidence.

### Important limitations

This classifier is a fast rule-based routing layer, not a general-purpose semantic classifier and not a medical/legal safety system. It can miss synonyms, multilingual queries, implicit domains and ambiguous phrasing. Low-confidence or unknown queries proceed to retrieval rather than being rejected. Extend the domain taxonomy and term dictionaries for your corpus; evaluate false-positive and false-negative rates. For complex or low-confidence queries, a structured LLM classifier can be added as a bounded fallback, but its output must never override tenant/ACL filters or evidence checks. Domain labels are metadata supplied during ingestion; validate them and restrict who can assign them. Query intent classification does not establish that a response is factually correct.

The exact-query Redis cache is now namespaced by tenant and effective metadata filters to avoid cross-tenant or cross-domain answer reuse. Session chats continue to bypass the global answer cache.

## Design

Local/private: PDFs, OCR, image OCR, native table extraction, PII masking, chunking, CPU embeddings, Qdrant, SQLite FTS5 BM25 retrieval and Redis cache. Remote: only the masked user question and selected masked passages are sent to the configured hosted LLM. If even excerpts may not leave, use a local LLM; 2–5 second p95 is unlikely on CPU.

## Extraction coverage

- PyMuPDF native text and page layout.
- Tesseract full-page OCR fallback for scanned/image-only pages.
- OCR for extractable embedded raster images.
- PyMuPDF table extraction where supported; tables serialized as pipe-delimited text.
- PII masking with Presidio before indexing.
- Repeated edge-line cleanup and page-level provenance.

For complex diagrams, plots, handwriting and visually meaningful figures, OCR alone is not enough. Add a vision-language captioning stage in a controlled environment and store its generated description as a separately labelled `image_description` chunk. For complex tables, use Docling or a layout-aware parser and validate row/column structure.

## Chunking and embedding

Starting configuration: 650 model tokens, 100 overlap, page/section boundaries retained, table content labelled, page provenance kept. This is a baseline to evaluate, not universally optimal. For long documents, use parent-child retrieval: index 250–400-token child chunks and fetch the parent section for answer context. The included starter uses page-local token chunks. BGE-small-en-v1.5 is a fast CPU baseline; benchmark BGE-base or BGE-M3 if recall is inadequate. Do not change embedding model without rebuilding the collection.

## Retrieval and reranking

Dense retrieval uses Qdrant HNSW for top-50 candidates. Independent full-corpus lexical retrieval uses SQLite FTS5 BM25 for top-50 candidates. Reciprocal Rank Fusion merges both lists, then FlashRank TinyBERT reranks the best 25 and selects the final 7 passages. For larger corpora or heavy write concurrency, replace SQLite FTS5 with OpenSearch or Tantivy while preserving the retrieval interface.

## Cache

Redis exact normalized-query cache stores only verified answers. Key includes `CACHE_VERSION`; bump it after document corpus, prompt, model, embedding or access-policy changes. This starter deliberately avoids approximate semantic cache reuse: semantically similar questions can have different constraints, dates or requested scope. Add vector semantic cache only with a high threshold, tenant/user/document-scope isolation, PII-safe keys, and a second relevance check. Never cache unverified answers. For personalized or permission-scoped corpora, include user/ACL scope in the cache key.

## Agentic correctness

1. Retrieve and rerank.
2. Generate a grounded draft with page citations.
3. Deterministically validate that cited filename/page pairs exist in retrieved evidence.
4. If citations fail, perform one bounded repair pass.
5. Cache only when validation passes.

This is a citation-integrity check, not proof of factual entailment. Production should add a separate verifier model or NLI/LLM judge that checks each claim against evidence, flags contradictions and grades answer completeness. If failed, perform one targeted retrieval rewrite or ask for clarification; cap retries to avoid latency/cost loops. Return an explicit unverified status rather than presenting a failed answer as correct.

## Run

```bash
cp .env.example .env
# Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL and API_KEY.
# Generate API_KEY with: openssl rand -hex 32
docker compose up -d --build
mkdir -p data/pdfs
# Copy PDF files into data/pdfs/
docker compose exec api python -m rag.ingest --dir /app/data/pdfs
```

API:
```bash
curl -X POST http://127.0.0.1:8000/chat \
 -H 'Authorization: Bearer YOUR_API_KEY' \
 -H 'Content-Type: application/json' \
 -d '{"question":"What does the policy say about retention?"}'
```

## Latency and capacity

For 5–20 concurrent users with CPU-only private infrastructure:
- Keep CPU embedding model warm; avoid per-request model loading.
- OCR and ingestion are offline jobs, never query-time.
- Use 1 API worker initially; multiple workers duplicate model memory.
- Use a semaphore and provider-side concurrency/queue controls.
- Use cloud GPU inference with short context and 200–350 token outputs.
- Redis exact cache hit bypasses retrieval and generation.
- Measure p50/p95/p99 with 5, 10 and 20 concurrent users. The 2–5 second p95 is a target, not a guarantee.
- A cross-encoder on CPU may be a bottleneck; benchmark it. FlashRank TinyBERT is selected as a lightweight option.

## Security / data quality production checklist

- TLS reverse proxy/VPN; no public Qdrant or Redis ports.
- Disk encryption, encrypted backups, restore drills.
- Use a secrets manager; rotate API keys.
- Provider no-training/no-retention and approved region terms must be verified contractually.
- PII masking is imperfect. Add domain-specific recognizers, tests and human review for sensitive corpora. Keep originals access-controlled; do not store raw text in logs/cache.
- Presidio's default recognizers are not a complete global PII detector. This sample is English-focused.
- Validate page counts, empty extraction ratio, OCR confidence, duplicate pages, malformed PDFs and unsupported files. Quarantine low-quality documents for review.
- Add ACL filters to every dense/lexical retrieval and cache key before multi-user/multi-tenant deployment.
- Current API key is a shared service credential, not end-user authentication.
- Validate citations and factual entailment separately.
- Use a document manifest with content hashes, extraction status, chunk count and index version. Atomic ingestion/reconciliation is needed for production updates.

## Next production components

Add OpenSearch/Tantivy BM25; Docling layout-aware parsing; parent-child chunk storage; image captioning for charts/diagrams; semantic Redis cache with ACL isolation; verifier model/NLI; document ACLs; rate limiting; ingestion queue; and load/evaluation suite.

## User sessions, conversation memory and scaling (added)

### Identity and session ownership

- `POST /sessions` creates a user-owned conversation; `GET /sessions` lists that user's active sessions; `GET /sessions/{id}/messages` returns that session's history.
- `POST /sessions/{id}/chat` answers within a session. Session IDs are UUIDs, but are not treated as authorization: every read/write checks the authenticated user's ownership and expiry.
- End-user requests use a JWT access token from your identity provider. The API validates `sub` and `exp` using `JWT_SECRET`/`JWT_ALGORITHM`. The shared `API_KEY` is retained only as a service-admin credential for operational tasks. Do not distribute that key to end users.
- Configure your identity provider to issue signed tokens. Never trust a user ID supplied as a plain request header or body field. For production, use asymmetric JWT validation (for example RS256 with your IdP public key/JWKS) and validate issuer, audience and scopes.
- PostgreSQL stores session metadata, messages and rolling summaries. Sessions expire after `SESSION_TTL_DAYS`; add a scheduled cleanup job for expired rows and a retention policy appropriate to your privacy requirements.

### Context-window management

- Each turn loads the compact session summary plus recent turns. The last `HISTORY_RECENT_MESSAGES` messages are retained as detailed context, subject to `HISTORY_MAX_CHARS`.
- When history grows beyond `SUMMARY_TRIGGER_MESSAGES`, older unsummarized turns are compacted by the configured LLM into a short memory. The original messages remain in PostgreSQL for audit/history retrieval; the summary is only a prompt optimization.
- Retrieval receives the recent dialogue plus current question to improve follow-up resolution. The final generation prompt receives the summary, recent dialogue, and reranked PDF evidence.
- Prior assistant messages are context only, never source evidence. Material claims still need document citations. The starter uses a character budget as a lightweight guard; production should use the exact tokenizer for the selected model and reserve tokens for system prompt, evidence, output and safety margin.
- Session-scoped chats bypass the global exact-query cache to avoid cross-session/context leakage. If adding a session cache later, include user ID, session ID, corpus version and ACL scope in the cache key.

### Rate limiting

Redis fixed-window limits are configurable through `USER_RPM`, `SESSION_RPM` and `GLOBAL_RPM`. Defaults are 30 requests/user/minute, 12 requests/session/minute and 300 requests/global/minute. Requests exceeding a limit return HTTP 429 with a retry-after value. Tune these limits against model-provider quotas and actual user behavior; consider token-based quotas as well as request counts for production.

### Celery background processing

PDF ingestion can be queued with the admin-only endpoint `POST /admin/ingest/{filename}`. Celery workers consume the `ingestion` queue, so OCR and embedding do not occupy API request workers. Check task status through your Celery monitoring system (for example Flower, if deployed); do not expose the Redis broker or result backend publicly. Worker retries are bounded. For large corpora, add idempotent ingestion manifests, task deduplication, dead-letter handling and index reconciliation.

Scale API replicas behind a TLS reverse proxy/load balancer and scale ingestion workers independently. Example: `docker compose up -d --scale celery-worker=3`. Start with 2 worker processes per replica on CPU and benchmark memory/CPU before increasing concurrency. Redis is configured with `noeviction` so session/rate-limit keys are not silently evicted; monitor memory and configure capacity/alerts. Use a managed highly available PostgreSQL/Redis deployment for production.

### New API flow

```bash
# End-user token must be issued by your configured identity provider.
export TOKEN='YOUR_SIGNED_JWT'

# Create a session
curl -X POST http://127.0.0.1:8000/sessions \\
  -H "Authorization: Bearer $TOKEN"

# Chat in that session
curl -X POST http://127.0.0.1:8000/sessions/SESSION_UUID/chat \\
  -H "Authorization: Bearer $TOKEN" \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"Summarize the eligibility rules."}'

# List sessions
curl http://127.0.0.1:8000/sessions -H "Authorization: Bearer $TOKEN"
```

The existing `/chat` endpoint remains for stateless requests. Use `/sessions/{session_id}/chat` for conversation-aware behavior.

## Production caveats for the added features

This is still an implementation starter, not a security-audited or load-tested production deployment. JWT validation currently uses shared-secret HS256 and must be connected to your identity provider with issuer/audience checks (prefer JWKS/RS256 in production). The summarization call adds latency when compaction is triggered; move it to a background task only if you also implement a safe summary-version/locking strategy. Rate limiting is fixed-window (not a smooth token bucket) and applies at the API layer; enforce provider/model quotas too. The sample does not yet include user-specific document ACLs: add tenant/user filters to both Qdrant and lexical retrieval before allowing different users to access different document sets. Run migration, concurrency, ownership-isolation, retention, rate-limit and 5/10/20-user load tests before deployment.

## Offline evaluation and golden test pipeline (added)

A separate evaluation suite is available under `eval/`; it is not invoked by the live chat API. See [`eval/README.md`](eval/README.md) for dataset authoring, metrics, and experiment procedure.

Quick retrieval/routing evaluation (after indexing a test corpus):

```bash
python -m eval.run_eval --dataset eval/datasets/golden.template.jsonl --mode retrieval
```

For full generation plus RAGAS metrics, install the optional dependencies and use an expert-curated dataset:

```bash
pip install -r requirements-eval.txt
python -m eval.run_eval --dataset eval/datasets/golden.v1.jsonl --mode full --ragas
```

The included JSONL is an illustrative template only. Replace placeholder ground truths, reference passages and relevant chunk IDs with expert-reviewed labels before interpreting scores. Outputs are written to `eval/results/`. Metrics include routing/domain accuracy, Hit@5, MRR, nDCG@5, citation-integrity rate, and optional RAGAS faithfulness, answer relevancy, context precision, context recall and answer correctness. Evaluation runs offline against a staging/test corpus and may incur LLM judge costs.

## Web frontend (added)

A React + TypeScript + Vite frontend is included in `frontend/`. It uses the rose/pink, blush and charcoal palette from the supplied visual reference.

### Run with Docker Compose
Set `VITE_API_URL` at frontend build time if the browser should reach an API address other than `http://localhost:8000`. The default works for local development:
```bash
docker compose up --build -d
```
Open `http://localhost:3000`. The API Swagger UI remains at `http://localhost:8000/docs`.

### Run frontend locally for development
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`. The frontend expects the API at `http://localhost:8000`; set `VITE_API_URL` in a `frontend/.env` file to override it.

### Frontend capabilities and current limitations
- Bearer-token sign-in, session list/create, session history and session-aware chat.
- Domain and tag filters, answer verification status, latency, retrieval count and source citation display.
- Admin ingestion task submission for PDFs already copied into the backend `data/pdfs` folder.
- Evaluation command guidance and report location.
- The current backend does not expose a browser PDF upload or document-list endpoint, so the UI does not pretend to upload/list files. Copy PDFs to `data/pdfs` first, then queue ingestion from the Documents screen.
- JWT issuance/user registration is handled by your identity provider; the sign-in screen accepts an existing bearer token. It is not a password-based login system.
- For production, serve frontend and API behind HTTPS/reverse proxy, set a strict CORS allowlist, configure a real identity provider and avoid using the service admin key for ordinary users.


## NavGuru.bot frontend branding

The web application is branded as **NavGuru.bot**. The frontend uses:
- Primary magenta / deep pink: `#D11A5F` (`rgb(209, 26, 95)`)
- Soft pink background: `#FFE6EC` (`rgb(255, 230, 236)`)

The palette is defined in `frontend/src/style.css` and the browser title is configured in `frontend/index.html`.
