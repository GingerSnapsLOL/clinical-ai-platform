## Clinical AI Platform

Privacy-first clinical AI microservice stack for clinical question answering with grounded retrieval and risk scoring.

Core contract:

- Public API stays `POST /v1/ask` on `gateway-api` (`:8000`)
- Shared models live in `services/shared/schemas_v1.py`
- `trace_id` is propagated across all hops

## Current services

| Service | Port | Purpose |
|---|---:|---|
| `gateway-api` | `8000` | Public entrypoint. Accepts `trace_id` optional, generates one if missing, forwards to orchestrator. |
| `orchestrator` | `8010` | End-to-end flow orchestration and final `AskResponse` assembly. |
| `pii-service` | `8020` | PII redaction (`POST /v1/redact`). |
| `ner-service` | `8030` | Clinical entity extraction (`POST /v1/extract`). |
| `retrieval-service` | `8040` | Corpus ingest + retrieval over Qdrant (`POST /v1/retrieve`). |
| `scoring-service` | `8050` | Multi-target scoring (`POST /v1/score`). |
| `llm-service` | `8060` | Text generation endpoint used by orchestrator synthesis paths. |
| `frontend` | `3000` | Internal UI for ask flow. |

Stateful dependencies in compose:

- `postgres` (`5432`)
- `redis` (`6379`)
- `qdrant` (`6333`)

## Orchestrator behavior (as of now)

`POST /v1/ask` in `services/orchestrator/app/main.py` supports two internal paths:

1. **Supervised multi-agent pipeline** (`ORCHESTRATOR_SUPERVISOR_PIPELINE=true`)
   - Runs `SupervisorCoordinator.run()` with bounded deterministic steps:
     - clinical structuring
     - retrieval + scoring
     - relevance gate
     - evidence critic
     - safety
     - synthesis
   - Includes step logging with `trace_id`
   - Supports debug metadata via `ORCHESTRATOR_AGENT_DEBUG=true` or `user_context.debug`
   - Enforces max step budget (`MAX_WORKFLOW_STEPS_V1`) to prevent loops

2. **Legacy orchestrator path** (`ORCHESTRATOR_SUPERVISOR_PIPELINE=false`)
   - PII -> NER -> retrieval + scoring -> relevance gate -> answer generation
   - Optional internal agent runtime controlled by `ORCHESTRATOR_AGENT_MODE`

Response contract remains `AskResponse` (includes `answer`, `sources`, `entities`, `risk`, `warnings`, `citations`, timings).

## Scoring service startup fallback (new)

`scoring-service` now supports booting without a trained model artifact.

- If `/app/models/triage_severity/model.pkl` is missing and `SCORING_ALLOW_MOCK_MODEL=true` (default), loader uses a deterministic in-memory mock baseline model.
- This is intended for development/bootstrap before training.
- To enforce real artifacts in stricter environments:
  - set `SCORING_ALLOW_MOCK_MODEL=false`

## Quick start

```bash
cp .env.example .env
make up
```

Useful commands:

```bash
make logs
make ps
make health
make down
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

Frontend lives in `frontend/` and is used as an internal clinical testing console.

Stack:

- Next.js (App Router)
- TypeScript
- Tailwind CSS
- ESLint

Implemented pages and modules:

- `frontend/app/page.tsx` - landing page
- `frontend/app/ask/page.tsx` - main ask console
- `frontend/app/api/ask/route.ts` - proxy route to backend `/v1/ask`
- `frontend/components/*` - reusable UI panels/cards
- `frontend/lib/api.ts` and `frontend/lib/types.ts` - typed frontend API layer

Current UX capabilities:

- answer panel with loading/error states
- sources evidence cards (title, metadata, relevance, snippet)
- entities panel
- risk panel with low/medium/high badge and explanation
- diagnostics panel (total/retrieval/llm timings)
- trace/debug collapsible panel (`trace_id`, warnings, retrieval diagnostics)
- compare mode (Answer A vs Answer B with latency/source differences)
- one-click demo prompt prefills

Frontend environment:

```env
BACKEND_BASE_URL=http://localhost:8000
```

Run frontend locally:

```bash
cd frontend
npm install
node ./node_modules/next/dist/bin/next dev
```

Open:

- `http://localhost:3000`
- `http://localhost:3000/ask`

## Key environment flags

| Variable | Default | Notes |
|---|---|---|
| `ORCHESTRATOR_SUPERVISOR_PIPELINE` | `false` | Enables `SupervisorCoordinator` pipeline branch. |
| `ORCHESTRATOR_AGENT_DEBUG` | `false` | Adds richer agent step logging in supervised mode. |
| `ORCHESTRATOR_AGENT_MODE` | `false` | Enables internal agent-runtime synthesis on legacy path. |
| `ORCHESTRATOR_CACHE_ENABLED` | `false` | Enables Redis caching in orchestrator. |
| `SCORING_ALLOW_MOCK_MODEL` | `true` | Allows scoring startup with deterministic mock model if artifact missing. |

Service URL defaults used by orchestrator:

- `PII_SERVICE_URL=http://pii-service:8020`
- `NER_SERVICE_URL=http://ner-service:8030`
- `RETRIEVAL_SERVICE_URL=http://retrieval-service:8040`
- `SCORING_SERVICE_URL=http://scoring-service:8050`
- `LLM_BASE_URL=http://llm-service:8060`

## Infra note

Infrastructure currently runs from root `docker-compose.yml`.
The `infra/` directory is reserved for future IaC and ops assets (compose overlays, Grafana, Prometheus, provisioning).

## Development and tests

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

## References

- `docs/architecture.md`
- `docs/api_contracts.md`
- `docs/roadmap.md`
- `docs/implementation_plan.md`
