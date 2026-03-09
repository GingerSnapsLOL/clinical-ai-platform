# Clinical AI Platform

Production-style, privacy-first Clinical AI system. PII redaction, medical NER, RAG over Qdrant with cross-encoder reranking, tabular risk scoring, and deterministic answer synthesis. Optional web enrichment (Hybrid mode) and local LLM inference planned for later milestones.

## Stack

- **Python 3.11+**, **uv** for dependency management
- **FastAPI** for all Python services, **Pydantic v2** for typed contracts
- **httpx** for inter-service calls, **pytest** for tests
- **Docker Compose** for local deployment
- **Presidio** (PII), **SciSpaCy** (biomedical NER), **sentence-transformers** (embeddings + reranking), **Qdrant** (vector DB)

## Services

| Service           | Port | Description                                          |
|-------------------|------|------------------------------------------------------|
| gateway-api       | 8000 | Single entrypoint, forwards to orchestrator          |
| orchestrator      | 8010 | Pipeline orchestration, query enrichment, synthesis  |
| pii-service       | 8020 | PII detect + anonymize (Presidio, spaCy en_core_web_lg) |
| ner-service       | 8030 | Medical entity extraction (SciSpaCy BC5CDR)          |
| retrieval-service | 8040 | RAG (Qdrant), cross-encoder reranking, top 3 passages |
| scoring-service   | 8050 | Risk scoring + SHAP                                  |

**Infrastructure:** postgres (5432), redis (6379), qdrant (6333).

## Quick Start

1. Copy env and start:

```bash
cp .env.example .env
docker compose up --build
```

2. Ingest demo documents (run once before querying):

```bash
python scripts/ingest_demo.py
```

3. Call `/v1/ask`:

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "strict",
    "note_text": "Patient John Doe has hypertension treated with lisinopril",
    "question": "What are the treatment risks?"
  }'
```

## Health Checks

| Service   | URL                            |
|-----------|--------------------------------|
| gateway   | http://localhost:8000/health   |
| orchestrator | http://localhost:8010/health |
| pii       | http://localhost:8020/health   |
| ner       | http://localhost:8030/health   |
| retrieval | http://localhost:8040/health   |
| scoring   | http://localhost:8050/health   |

## Pipeline (M3 — End-to-End)

Request flow: **gateway → orchestrator → services**

1. **PII redaction** — Presidio anonymizes note text (PERSON, PHONE, EMAIL → tokens)
2. **Clinical NER** — SciSpaCy extracts entities (DISEASE, CHEMICAL, etc.), normalized
3. **Retrieval** — Enriched query (question + entities + note summary) → Qdrant search → cross-encoder rerank → top 3 passages
4. **Risk scoring** — Scoring-service returns label (low/medium/high) and feature contributions
5. **Response assembly** — Deterministic synthesis: summary, key risks, recommended monitoring; sources and citations from retrieval

## API Overview

### Gateway `/v1/ask`

Single entrypoint. Accepts `note_text`, `question`, optional `mode` and `trace_id`. Returns:

- `answer` — Structured text (Summary, Key risks, Recommended monitoring)
- `sources` — Top 3 reranked passages (source_id, snippet, score, metadata)
- `citations` — Unique source references (source_id, title, url)
- `entities` — Extracted medical entities from NER
- `risk` — Score, label, explanation from scoring-service

### PII `/v1/redact`

- Input: `trace_id`, `text`
- Output: `redacted_text`, `spans` (type, start, end, replacement, confidence)

### NER `/v1/extract`

- Input: `trace_id`, `text`
- Output: `entities` (type, text, start, end, confidence); types normalized (e.g. DRUG→CHEMICAL)

### Retrieval `/v1/retrieve`

- Input: `trace_id`, `query`, `top_k`, `top_n`, `rerank`
- Output: `passages` (source_id, text, score, metadata with doc_id, title, source)
- With `rerank=true`: cross-encoder (ms-marco-MiniLM-L-6-v2), top 3 passages; deduplicated by text

### Retrieval `/v1/ingest`

- Input: `documents` (doc_id, text, metadata)
- Chunks text, embeds with all-MiniLM-L6-v2, upserts into Qdrant

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ingest_demo.py` | Ingest `examples/clinical_docs.json` into retrieval-service |
| `scripts/eval_retrieval.py` | Test retrieval: `python scripts/eval_retrieval.py --query "hypertension treatment"` |
| `scripts/demo_m1.py` | Call gateway `/v1/ask`; optional `--payload examples/ask_request.json` |

## Development

### Dependencies

```bash
uv sync
```

### Tests

```bash
make test
```

Or per service:

```bash
PYTHONPATH=.:services/pii-service uv run pytest services/pii-service/tests -v
PYTHONPATH=.:services/ner-service uv run pytest services/ner-service/tests -v
PYTHONPATH=.:services/retrieval-service uv run pytest services/retrieval-service/tests -v
# etc.
```

### Make Targets

| Target  | Command           | Description                    |
|---------|-------------------|--------------------------------|
| `make up` | `docker compose up -d` | Start stack                 |
| `make rebuild` | `lock + docker compose up -d --build` | Rebuild and start |
| `make down` | `docker compose down -v` | Stop and remove volumes  |
| `make logs` | `docker compose logs -f --tail=200` | Follow logs        |
| `make health` | curl health endpoints | Quick health check    |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ORCHESTRATOR_URL` | Orchestrator base URL (Docker) | `http://orchestrator:8010` |
| `PII_SERVICE_URL` | PII service base URL | `http://pii-service:8020` |
| `NER_SERVICE_URL` | NER service base URL | `http://ner-service:8030` |
| `RETRIEVAL_SERVICE_URL` | Retrieval service base URL | `http://retrieval-service:8040` |
| `SCORING_SERVICE_URL` | Scoring service base URL | `http://scoring-service:8050` |
| `QDRANT_HOST`, `QDRANT_PORT` | Qdrant connection | `qdrant`, `6333` |
| `RETRIEVAL_URL` | Used by scripts (localhost) | `http://localhost:8040` |

## Layout

```
services/
  gateway-api/    — single API entrypoint
  orchestrator/   — pipeline, query enrichment, synthesis
  pii-service/    — Presidio
  ner-service/    — SciSpaCy BC5CDR
  retrieval-service/ — Qdrant, embeddings, cross-encoder rerank
  scoring-service/
  shared/         — schemas_v1.py, http_client, logging_util
examples/         — ask_request.json, clinical_docs.json
scripts/          — ingest_demo.py, eval_retrieval.py, demo_m1.py
```

## Modes

- **Strict (default)**: No internet; only local KB (Qdrant, guidelines)
- **Hybrid** (planned): Web search/page fetch; queries de-identified; allowlist domains

## Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | ✅ | Microservice architecture, `/health`, trace_id |
| M2 | ✅ | Retrieval pipeline (ingest, retrieve, rerank, top 3) |
| M3 | ✅ | Clinical NER (SciSpaCy), PII (Presidio), full pipeline |
| M4 | ⬜ | LLM answer synthesis (planned) |

**Note:** Answer synthesis is currently deterministic (summary + risks + monitoring from passages). LLM integration is not yet implemented.

## Docs

- `PROJECT_SPEC.md` — architecture, milestones, data
- `AI_RULES.md` — project conventions
