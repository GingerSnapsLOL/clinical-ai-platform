# Clinical AI Platform

Production-style, privacy-first Clinical AI system. PII redaction, medical NER, RAG over Qdrant, optional web enrichment (Hybrid mode), local LLM inference, tabular risk scoring with SHAP, LangGraph orchestration.

## Stack

- **Python 3.11+**, **uv** for dependency management
- **FastAPI** for all Python services, **Pydantic v2** for typed contracts
- **httpx** for inter-service calls, **pytest** for tests
- **Docker Compose** for local deployment

## Services

| Service           | Port | Description                          |
|-------------------|------|--------------------------------------|
| gateway-api       | 8000 | Auth, audit, rate limits, single API |
| orchestrator      | 8010 | LangGraph pipeline, budgets          |
| pii-service       | 8020 | PII detect + anonymize               |
| ner-service       | 8030 | Medical entity extraction            |
| retrieval-service | 8040 | RAG (Qdrant) + rerank                |
| scoring-service   | 8050 | Risk scoring + SHAP                  |

Infrastructure: **postgres** (5432), **redis** (6379), **qdrant** (6333).

## Quick start

1. Copy env and start:

```bash
cp .env.example .env
docker compose up --build
```

2. Health checks:

| Service   | URL                            |
|-----------|--------------------------------|
| gateway   | http://localhost:8000/health   |
| orchestrator | http://localhost:8010/health |
| pii       | http://localhost:8020/health   |
| ner       | http://localhost:8030/health   |
| retrieval | http://localhost:8040/health   |
| scoring   | http://localhost:8050/health   |

3. Test `/v1/ask`:

```bash
python scripts/demo_m1.py
# or with a payload file:
python scripts/demo_m1.py --payload examples/ask_request.json
```

Or with curl (use a file to avoid shell quoting):

```bash
curl -X POST http://localhost:8000/v1/ask -H "Content-Type: application/json" -d @examples/ask_request.json
```

## Development

### Dependencies (uv)

```bash
uv sync
```

### Tests

```bash
make test
# or run per service:
uv run pytest services/shared/tests -v
PYTHONPATH=.:services/gateway-api uv run pytest services/gateway-api/tests -v
# ... same for orchestrator, pii-service, ner-service, retrieval-service, scoring-service
```

## Modes

- **Strict (default)**: No internet; only local KB (Qdrant, guidelines).
- **Hybrid**: Web search/page fetch allowed; queries de-identified; allowlist domains; content untrusted and cited.

## Layout

- `services/` — gateway-api, orchestrator, pii, ner, retrieval, scoring (each with `app/`, `tests/`, `Dockerfile`, `pyproject.toml`)
- `services/shared/` — typed schemas (`schemas_v1.py`), http client, logging
- `examples/` — `ask_request.json`, `ask_response.json`
- `scripts/` — `demo_m1.py`, `demo_m1.sh`
- `data/`, `eval/`, `infra/` — KB sources, evaluation, optional observability

## Docs

- `PROJECT_SPEC.md` — architecture, milestones, data
