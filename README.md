## Clinical AI Platform

[![CI](https://github.com/GingerSnapsLOL/clinical-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/GingerSnapsLOL/clinical-ai-platform/actions/workflows/ci.yml)

Privacy-first clinical AI microservice stack for clinical question answering with grounded retrieval and risk scoring.

Core contract:

- Public API: `POST /v1/ask` on `gateway-api` (`:8000`)
- Shared typed models: `services/shared/schemas_v1.py`
- `trace_id` is propagated across all hops (`X-Trace-Id` + body fields)

## Current architecture (implemented)

Pattern:

- Dockerized FastAPI microservices
- Shared Pydantic contracts
- Structured JSON logs with propagated trace context

Data flow (strict mode):

1. Client calls `gateway-api` `/v1/ask`
2. `gateway-api` validates input and generates `trace_id` when missing
3. `orchestrator` executes:
   - PII redaction (`pii-service`)
   - NER extraction (`ner-service`)
   - retrieval + scoring in parallel (`retrieval-service` and `scoring-service`)
   - synthesis via `llm-service` or deterministic fallback
4. `orchestrator` returns typed `AskResponse` with sources/entities/risk/warnings/citations/timings

## Services and ports

| Service | Port | Endpoints | Purpose |
|---|---:|---|---|
| `gateway-api` | `8000` | `GET /health`, `POST /v1/ask` | Public entrypoint and request validation/proxying |
| `orchestrator` | `8010` | `GET /health`, `POST /v1/ask` | End-to-end ask orchestration |
| `pii-service` | `8020` | `GET /health`, `POST /v1/redact` | PII redaction |
| `ner-service` | `8030` | `GET /health`, `POST /v1/extract` | Clinical/biomedical entity extraction |
| `retrieval-service` | `8040` | `GET /health`, `POST /v1/retrieve`, `POST /v1/ingest` | Qdrant ingest and retrieval |
| `scoring-service` | `8050` | `GET /health`, `POST /v1/score` | Multi-target risk scoring |
| `llm-service` | `8060` | `GET /health`, `POST /v1/generate` | Local text generation |
| `frontend` | `3000` | UI routes (`/`, `/ask`) | Internal testing UI |

Stateful dependencies in compose:

- `postgres` (`5432`)
- `redis` (`6379`)
- `qdrant` (`6333`)

## Orchestrator modes and agent paths

`POST /v1/ask` in `services/orchestrator/app/main.py` supports two internal branches:

1. **Supervised pipeline** (`ORCHESTRATOR_SUPERVISOR_PIPELINE=true`)
   - Entry: `services/orchestrator/app/agent_pipeline.py`
   - Coordinator: `SupervisorCoordinator` in `services/orchestrator/app/agents/coordinator.py`
   - Fixed step sequence (no unbounded loops) with step-level telemetry

2. **Legacy direct pipeline** (`ORCHESTRATOR_SUPERVISOR_PIPELINE=false`)
   - Direct PII -> NER -> retrieval/scoring -> synthesis flow in `main.py`
   - Optional legacy linear runtime when `ORCHESTRATOR_AGENT_MODE=true`:
     `services/orchestrator/agent_runtime.py`, `agent_nodes.py`, `agent_state.py`

Supervised debug options:

- `ORCHESTRATOR_AGENT_DEBUG=true`
- or pass `user_context.debug` / `user_context.agent_debug`

## API contracts summary

Source of truth: `services/shared/schemas_v1.py`. The gateway accepts the same ask shape with **`trace_id` optional** and fills it when omitted (see `services/gateway-api/app/main.py`).

Conventions:

- `trace_id` exists on all core request/response models
- `status` is shared: `"ok"` or `"error"`
- errors use `ErrorInfo { code, message, details }`

Main contracts:

- `POST /v1/ask`:
  - request: `AskRequest` (`mode` supports `"strict"` and `"hybrid"`)
  - response: `AskResponse` (`answer`, `sources`, `entities`, `risk`, `warnings`, `citations`, timings)
- `POST /v1/redact`: `RedactRequest` -> `RedactResponse`
- `POST /v1/extract`: `ExtractRequest` -> `ExtractResponse`
- `POST /v1/retrieve`: `RetrieveRequest` -> `RetrieveResponse`
- `POST /v1/score`: `ScoreRequest` -> `ScoreResponse`

LLM contract:

- `llm-service` defines `GenerateRequest`/`GenerateResponse` locally in `services/llm-service/app/main.py`
- orchestrator uses `services/shared/llm_client.py` with `LLM_BASE_URL`

## Scoring behavior (current)

`scoring-service` is no longer hardcoded to a fixed stub response.

- It loads per-target model packages from `models/<target_id>/`
- If `triage_severity` artifact is missing and `SCORING_ALLOW_MOCK_MODEL=true` (default), it starts with deterministic mock baseline
- Set `SCORING_ALLOW_MOCK_MODEL=false` to require real model artifacts at startup

## Quick start

```bash
cp .env.example .env
make up
```

`make up` runs `setup` (creates `.env` from `.env.example` if missing), `uv lock` in each backend service, builds the shared **`clinical-ai-base`** image from `infra/clinical-ai-base.Dockerfile`, builds all Compose images, then starts the stack. Expect Docker work on first run.

Optional: load retrieval data after services are healthy:

```bash
INIT=1 make up
```

(or `make ingest` once Qdrant is up — see `Makefile`).

Useful commands:

```bash
make logs
make ps
make health
make down
make ci          # `make all` + `make test` (no GitHub Actions in-repo)
```

Rebuild only:

```bash
make build
docker compose up -d
```

## Health checks

```bash
curl -fsS http://localhost:8000/health && echo " gateway"
curl -fsS http://localhost:8010/health && echo " orchestrator"
curl -fsS http://localhost:8020/health && echo " pii"
curl -fsS http://localhost:8030/health && echo " ner"
curl -fsS http://localhost:8040/health && echo " retrieval"
curl -fsS http://localhost:8050/health && echo " scoring"
curl -fsS http://localhost:8060/health && echo " llm"
```

## Ask API example

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "strict",
    "note_text": "65-year-old with hypertension and diabetes on ACE inhibitor.",
    "question": "What monitoring should be prioritized?"
  }'
```

## Frontend (Next.js)

Frontend lives in `frontend/` and is an internal clinical testing console.

Implemented modules:

- `frontend/app/page.tsx` (landing)
- `frontend/app/ask/page.tsx` (main ask console)
- `frontend/app/api/ask/route.ts` (proxy to backend `/v1/ask`)
- `frontend/components/*` (UI panels/cards)
- `frontend/lib/api.ts`, `frontend/lib/types.ts` (typed frontend API layer)

Frontend env (see `frontend/app/api/ask/route.ts`): the API route uses **`BACKEND_BASE_URL`**, falling back to **`NEXT_PUBLIC_API_URL`** if unset.

Local dev (browser talks to your machine’s gateway):

```env
BACKEND_BASE_URL=http://localhost:8000
```

Docker Compose sets `NEXT_PUBLIC_API_URL=http://gateway-api:8000` for the containerized frontend.

Run locally:

```bash
cd frontend
npm install
npm run dev
```

Open:

- `http://localhost:3000`
- `http://localhost:3000/ask`

## Key environment flags

Canonical list with defaults: **`.env.example`**. Below are the main knobs not obvious from variable names alone.

| Variable | Default | Notes |
|---|---|---|
| `ORCHESTRATOR_SUPERVISOR_PIPELINE` | `false` | Enables supervised coordinator pipeline |
| `ORCHESTRATOR_AGENT_DEBUG` | `false` | Enables richer supervised step debug logs |
| `ORCHESTRATOR_AGENT_MODE` | `false` | Enables legacy agent runtime on non-supervised branch |
| `ORCHESTRATOR_CACHE_ENABLED` | `false` | Enables Redis caching in orchestrator |
| `ORCHESTRATOR_RETRIEVAL_CACHE_TTL_SEC` | `300` | Retrieval cache TTL when caching is on |
| `ORCHESTRATOR_ANSWER_CACHE_TTL_SEC` | `900` | Answer cache TTL when caching is on |
| `ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE` | `true` | When true, weak retrieval can short-circuit to an insufficient-data style path (tune scores below) |
| `ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE` | `1.0` | Min top retrieval score when gate is on (cross-encoder logits when `rerank=true`) |
| `ORCHESTRATOR_RETRIEVAL_MIN_TOP_SNIPPET_CHARS` | `24` | Min snippet length for gate |
| `SCORING_ALLOW_MOCK_MODEL` | `true` | Allows scoring startup fallback when model artifact is missing |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins for gateway CORS |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Used by `llm-service` in Compose when set |

Service URL defaults (Docker network hostnames):

- `PII_SERVICE_URL=http://pii-service:8020`
- `NER_SERVICE_URL=http://ner-service:8030`
- `RETRIEVAL_SERVICE_URL=http://retrieval-service:8040`
- `SCORING_SERVICE_URL=http://scoring-service:8050`
- `ORCHESTRATOR_URL=http://orchestrator:8010` (gateway → orchestrator)
- `LLM_BASE_URL=http://llm-service:8060`

## Development and tests

### CI (GitHub Actions)

On every push and pull request to `main`, [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs:

- **Ruff** on `services/` (syntax / import / basic style issues)
- **Mypy** on `services/shared/` (see [`mypy.ini`](mypy.ini); contract-focused)
- **`make test`** — `pytest` for `services/shared/tests` and each microservice’s `tests/`
- **`docker compose build`** — full image build after `make lock`

```bash
uv sync
make test
```

Single-service example:

```bash
cd services/orchestrator
uv sync
PYTHONPATH=$(pwd)/../.. uv run pytest tests -v --tb=short
```

## Project status and gaps

What is strong now:

- End-to-end strict RAG ask flow is implemented
- Retrieval pipeline and ingest scripts are functional
- Frontend exists and is wired to backend gateway
- Shared contracts and trace logging are in place

Known gaps:

- `mode="hybrid"` exists in schema but full web-enrichment tool path is not implemented
- Postgres is in compose but not primary runtime storage for audit/traces
- Security hardening (auth/rate limiting) is minimal at gateway level
- CI is GitHub Actions (see above); local `make ci` still runs `make all` + `make test` without Docker
- No separate long-form handbook under `docs/`; this README, `.env.example`, and `services/shared/schemas_v1.py` are the maintained references

## Roadmap (target architecture)

Future/aspirational items:

- Hybrid web enrichment with de-identification and allowlisted domains
- Policy-aware orchestration and stronger budgets/tool gating
- Richer evaluation pipelines (retrieval/NER/scoring) as quality gates
- Production hardening (secrets, observability stack, staged deployment)

Milestone direction:

- M0-M4 largely represented in current codebase
- M5+ (hybrid web tooling, broader monitoring/eval/deployment maturity) remains planned

## Infra

- **Compose**: root `docker-compose.yml` defines all runtime services.
- **Base image**: `infra/clinical-ai-base.Dockerfile` builds `clinical-ai-base`, which backend service images extend (`docker-compose.yml` `base:` service; `make base-image` / `make up`).

Additional ops assets (overlays, observability stacks, IaC) can live under `infra/` as the project grows.

