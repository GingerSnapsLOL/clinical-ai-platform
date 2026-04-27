## Clinical AI Platform

[![CI](https://github.com/GingerSnapsLOL/clinical-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/GingerSnapsLOL/clinical-ai-platform/actions/workflows/ci.yml)

Privacy-first clinical AI microservice stack for grounded question answering over clinical notes.

Core public contract:

- Public API: `POST /v1/ask` on `gateway-api` (`http://localhost:8000`)
- Shared schemas: `services/shared/schemas_v1.py`
- `trace_id` is propagated across services (`X-Trace-Id` + body fields)

---

## What is implemented now

- Dockerized FastAPI services with shared Pydantic contracts
- End-to-end strict ask flow (PII -> NER -> retrieval -> synthesis)
- Optional scoring step with deterministic rule-based logic (no fake ML)
- Frontend clinical workspace (`/ask`) with user mode + dev mode
- Structured logging and best-effort trace persistence

---

## Services and ports

| Service | Port | Main endpoints | Role |
|---|---:|---|---|
| `gateway-api` | `8000` | `GET /health`, `POST /v1/ask` | Public entrypoint and validation/proxy |
| `orchestrator` | `8010` | `GET /health`, `POST /v1/ask` | End-to-end workflow |
| `pii-service` | `8020` | `GET /health`, `POST /v1/redact` | PII redaction |
| `ner-service` | `8030` | `GET /health`, `POST /v1/extract` | Entity extraction |
| `retrieval-service` | `8040` | `GET /health`, `POST /v1/retrieve`, `POST /v1/ingest` | Retrieval + ingest |
| `scoring-service` | `8050` | `GET /health`, `POST /v1/score` | Rule-based triage scoring |
| `llm-service` | `8060` | `GET /health`, `POST /v1/generate` | Text generation |
| `frontend` | `3000` | `/`, `/ask`, `/chat` | Internal UI |

Stateful dependencies:

- `postgres` (`5432`)
- `redis` (`6379`)
- `qdrant` (`6333`)

---

## Ask flow (`POST /v1/ask`)

### Gateway

- Validates request and forwards to orchestrator
- Fills `trace_id` when omitted
- Returns `400` on request validation errors
- Validates orchestrator response shape before returning to client

### Orchestrator

Two internal execution paths:

1. **Supervised pipeline** (`ORCHESTRATOR_SUPERVISOR_PIPELINE=true`)
   - Entry: `services/orchestrator/app/agent_pipeline.py`
   - Coordinator: `services/orchestrator/app/agents/coordinator.py`
2. **Legacy direct pipeline** (`ORCHESTRATOR_SUPERVISOR_PIPELINE=false`)
   - Entry: `services/orchestrator/app/main.py`
   - Sequence: PII -> NER -> retrieval (+ optional scoring) -> synthesis

Behavior details (current):

- Retrieval and scoring run in parallel after NER
- Scoring is **optional**:
  - called only when relevant entities exist
  - scoring failures add warnings and do not stop answer generation
- Retrieval relevance gate can short-circuit answer to `Insufficient data`
- LLM failures use deterministic fallback synthesis (with warning)
- Diagnostics are returned only when debug is enabled (`user_context.debug` / env flags)
- Ask traces are persisted best-effort to Postgres when trace DB is enabled

---

## Scoring behavior (current, honest logic)

Scoring service (`services/scoring-service`) is deterministic and rule-based:

- `chest pain` -> `high`
- `fever` + `cough` -> `medium`
- otherwise -> `low`
- insufficient input -> `risk_available=false`, label `insufficient_data`

Response includes:

- `risk_available`
- `label`
- `confidence`
- narrative explanation (string)
- optional structured contributions

Orchestrator maps this into `risk_block` and frontend UI:

- if unavailable: shows “Risk assessment unavailable (insufficient data)”
- otherwise: shows label, confidence, and explanation

---

## Quick start

```bash
cp .env.example .env
make up
```

What `make up` does:

1. `setup` (`.env` bootstrap + `uv sync`)
2. `lock` (runs `uv lock` in each backend service)
3. `base-image` (builds `clinical-ai-base`)
4. `build` (compose build)
5. `docker compose up -d`

Optional ingest after startup:

```bash
INIT=1 make up
```

or:

```bash
make ingest
```

Useful commands:

```bash
make ps
make logs
make health
make build
make down
make test
```

---

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

---

## API example

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "strict",
    "note_text": "65-year-old with hypertension and diabetes on ACE inhibitor.",
    "question": "What monitoring should be prioritized?"
  }'
```

---

## Frontend (`frontend/`)

Primary route: `http://localhost:3000/ask`

Current UX:

- **User mode (default)**:
  - Answer (structured clinical-style sections)
  - Risk block
  - Simplified sources
- **Dev mode (toggle)**:
  - Diagnostics
  - Trace/debug blocks
  - Raw metadata sections

Extra UX behavior:

- Confidence indicator near answer (High / Medium / Low from grounding score)
- Weak-grounding badge
- Explicit fallback warning when deterministic fallback was used

Local frontend run:

```bash
cd frontend
npm install
npm run dev
```

Env for API proxy route:

- preferred: `BACKEND_BASE_URL`
- fallback: `NEXT_PUBLIC_API_URL`

---

## Important environment flags

Canonical defaults are in `.env.example`.

Key orchestrator flags:

- `ORCHESTRATOR_SUPERVISOR_PIPELINE` (default `false`)
- `ORCHESTRATOR_AGENT_MODE` (legacy runtime toggle, default `false`)
- `ORCHESTRATOR_AGENT_DEBUG` (debug logging toggle)
- `ORCHESTRATOR_CACHE_ENABLED` (Redis cache toggle)
- `ORCHESTRATOR_RETRIEVAL_CACHE_TTL_SEC`
- `ORCHESTRATOR_ANSWER_CACHE_TTL_SEC`
- `ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE`
- `ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE`
- `ORCHESTRATOR_RETRIEVAL_MIN_TOP_SNIPPET_CHARS`
- `ORCHESTRATOR_TRACE_DB_ENABLED`
- `ORCHESTRATOR_TRACE_DB_DSN` (optional; otherwise built from `POSTGRES_*`)

Service URL defaults (compose network):

- `PII_SERVICE_URL=http://pii-service:8020`
- `NER_SERVICE_URL=http://ner-service:8030`
- `RETRIEVAL_SERVICE_URL=http://retrieval-service:8040`
- `SCORING_SERVICE_URL=http://scoring-service:8050`
- `ORCHESTRATOR_URL=http://orchestrator:8010`
- `LLM_BASE_URL=http://llm-service:8060`

---

## Development and CI

Local tests:

```bash
uv sync
make test
```

CI (`.github/workflows/ci.yml`) currently runs:

- Ruff on `services/`
- Mypy on `services/shared/`
- `make test`
- `docker compose build`

---

## Known practical notes

- If Docker build fails at `uv sync --locked`, refresh lockfiles:

```bash
make lock
```

- If frontend Docker build fails at `npm run build`, inspect TypeScript error in logs (most recent failures were stale frontend types after backend contract changes).

- Retrieval service uses lazy model loading and has a compose healthcheck `start_period` to avoid false-unhealthy during startup.

---

## Repo references

- Shared schemas: `services/shared/schemas_v1.py`
- Orchestrator entrypoint: `services/orchestrator/app/main.py`
- Supervised pipeline: `services/orchestrator/app/agent_pipeline.py`
- Scoring rules: `services/scoring-service/app/rule_score.py`
- Compose topology: `docker-compose.yml`
- Task automation: `Makefile`

