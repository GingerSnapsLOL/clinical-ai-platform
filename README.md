## Clinical AI Platform

Privacy-first clinical AI backend focused on **PII redaction**, **biomedical NER**, **RAG over Qdrant with cross‑encoder reranking**, **multi-target risk scoring**, and **strictly grounded answer synthesis**.  
All services are Dockerized FastAPI microservices with shared typed contracts (`services/shared/schemas_v1.py`) and `trace_id` on every hop.

This README focuses on **how to run the system**. For design detail:

- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/api_contracts.md`
- `docs/implementation_plan.md`

---

## Services and current behavior

### Service map

| Service | Port | Role |
|--------|------|------|
| **Gateway API** | `:8000` | Public `POST /v1/ask`; validates payload; forwards to orchestrator. |
| **Orchestrator** | `:8010` | Full ask pipeline: PII → NER → **retrieval ∥ scoring** → relevance gate → answer synthesis (see below). Optional **Redis** answer/retrieval cache. |
| **PII Service** | `:8020` | `POST /v1/redact` — redacted text + spans. |
| **NER Service** | `:8030` | `POST /v1/extract` — entities on redacted text. |
| **Retrieval Service** | `:8040` | `POST /v1/ingest`, `POST /v1/retrieve` — Qdrant embeddings + reranked passages. |
| **Scoring Service** | `:8050` | `POST /v1/score` — rule-based signals; **multi-target** (`triage_severity` default; optional `targets=[...]`). |
| **LLM Service** | `:8060` | `POST /v1/generate` — text generation for orchestrator prompts. |

### Orchestrator ask flow

1. **PII** then **NER** on redacted note.
2. **Retrieval** and **scoring** run **in parallel** (same trace; scoring uses entities only).
3. **Relevance gate** may return `Insufficient data` without calling the LLM.
4. **Answer path**
   - **Default (`ORCHESTRATOR_AGENT_MODE=false`)**: single structured prompt + one `llm-service` call (`_build_llm_prompt`).
   - **Agent mode (`ORCHESTRATOR_AGENT_MODE=true`)**: internal **async runtime** (no LangGraph) — select top sources → draft (JSON: `answer`, `used_source_ids`) → verifier (JSON: `is_grounded`, `has_sufficient_evidence`, `problems`) → finalize. Same `AskResponse` schema; extra timing keys when the agent runs.

Agent modules live next to `app/` (e.g. `services/orchestrator/agent_state.py`, `agent_nodes.py`, `agent_runtime.py`) and are copied into the orchestrator image.

Design note: orchestration is explicit Python/async rather than LangChain/LangGraph for predictable latency, clear service boundaries, and simpler debugging.

### End-to-end path (conceptual)

`gateway-api` → `orchestrator` → (`pii-service` → `ner-service`) → (`retrieval-service` ∥ `scoring-service`) → `llm-service` (and/or internal agent nodes) → aggregated `AskResponse`.

### Contracts and traceability

- Shared schemas: `services/shared/schemas_v1.py`.
- Every hop carries **`trace_id`** (header + body where defined).
- **`GET /health`** on each service for smoke checks.

---

## Local run instructions

### Prerequisites

- **Python** 3.11+
- **uv** (optional for local dev; Docker builds use per-service locks)
- **Docker + Docker Compose**
- **make** (recommended)
- **Hardware**: `llm-service` may load **Qwen/Qwen2.5-7B-Instruct**; GPU helps; CPU-only is slow.

Installation hints: Python from `python.org` / `pyenv`; uv from [docs.astral.sh/uv](https://docs.astral.sh/uv/); Docker from [docs.docker.com](https://docs.docker.com/get-docker/); on Windows, prefer **WSL2** for `make` and paths.

### 1. Environment and stack

```bash
cp .env.example .env
make up           # setup, lock, base image, build service images, docker compose up -d
make logs
make down         # docker compose down -v (removes volumes)

# Docker images only (no compose up):
make all          # lock + base-image + build

# Full test suite after images build:
make ci           # make all && make test

# After stack is up: one-shot ingest (see Makefile — `all-ingest` chains `up` + `ingest`):
# make all-ingest
```

**Local uv (monorepo root)**

```bash
uv sync
```

### 2. Ingest retrieval corpus (when using Qdrant)

Qdrant should be reachable at `http://localhost:6333` (compose includes `qdrant`).

```bash
make ingest
# or, if datamix already exists:
uv run python scripts/ingest_qdrant.py
```

### 3. Smoke checks

**Health**

```bash
curl -fsS http://localhost:8000/health && echo " gateway"
curl -fsS http://localhost:8010/health && echo " orchestrator"
curl -fsS http://localhost:8020/health && echo " pii"
curl -fsS http://localhost:8030/health && echo " ner"
curl -fsS http://localhost:8040/health && echo " retrieval"
curl -fsS http://localhost:8050/health && echo " scoring"
curl -fsS http://localhost:8060/health && echo " llm"
```

**`POST /v1/ask` (via gateway, strict)**

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "strict",
    "note_text": "Patient has hypertension treated with lisinopril.",
    "question": "What are key monitoring considerations?"
  }'
```

With retrieval + LLM healthy you should see a non-empty `answer`, `entities`, `sources` (up to top_n passages), `risk` from scoring-service, and optional `timings` (including per-stage ms when populated).

### 4. Scripts (selection)

| Script | Purpose |
|--------|---------|
| `scripts/ingest_qdrant.py` | Embed `data/processed/datamix.jsonl` into Qdrant. |
| `scripts/eval_retrieval.py` | Retrieval sanity check. |
| `scripts/demo_m1.py` | Basic `/v1/ask` demo. |
| `scripts/demo_m4.py` | Rich demo (answer, citations, sources, entities, risk). |

---

## Frontend console (Next.js)

Internal testing UI under `frontend/` for `/v1/ask`.

- **Run**: `make frontend-install` / `make frontend-dev`, or `docker compose` includes a **frontend** service (~`:3000`).
- **Config**: `frontend/.env.local` — e.g. `BACKEND_BASE_URL=http://localhost:8000`
- **UI**: `http://localhost:3000`, ask flow at `/ask`

See `frontend/README.md` if present for component layout.

---

## Repository structure (high level)

```text
clinical-ai-platform/
  README.md
  Makefile
  docker-compose.yml
  pyproject.toml
  uv.lock
  docs/
    architecture.md
    roadmap.md
    api_contracts.md
    implementation_plan.md
  services/
    gateway-api/
    orchestrator/          # app/main.py + agent_*.py (runtime), tests/
    pii-service/
    ner-service/
    retrieval-service/
    scoring-service/      # multi-target scoring; features/rules under app/
    llm-service/
    shared/               # schemas, http_client, llm_client, logging
  frontend/
  scripts/
  examples/
  data/                  # processed/interim corpora (e.g. datamix.jsonl)
  infra/
```

---

## Environment variables (selection)

| Variable | Description | Typical default (compose) |
|----------|-------------|-----------------------------|
| `ORCHESTRATOR_URL` | Gateway → orchestrator | `http://orchestrator:8010` |
| `PII_SERVICE_URL` | Orchestrator → PII | `http://pii-service:8020` |
| `NER_SERVICE_URL` | Orchestrator → NER | `http://ner-service:8030` |
| `RETRIEVAL_SERVICE_URL` | Orchestrator → retrieval | `http://retrieval-service:8040` |
| `SCORING_SERVICE_URL` | Orchestrator → scoring | `http://scoring-service:8050` |
| `LLM_BASE_URL` | Orchestrator → LLM | `http://llm-service:8060` |
| `LLM_MODEL_NAME` | Model label / cache keys | `Qwen/Qwen2.5-7B-Instruct` |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant | `qdrant` / `6333` |
| `ORCHESTRATOR_AGENT_MODE` | `true` / `1` enables multi-step agent after relevance gate | `false` |
| `ORCHESTRATOR_CACHE_ENABLED` | Redis caching for orchestrator | often `false` locally |
| `REDIS_URL` or `REDIS_HOST` | Cache backend | compose sets `redis` |
| `RETRIEVAL_URL` | Used by scripts hitting retrieval locally | `http://localhost:8040` |

See `.env.example` for the full list.

---

## Development

### Dependencies

```bash
uv sync
```

### Tests

```bash
make test
```

Runs `services/shared/tests` and each service’s `tests/` with the correct `PYTHONPATH`. Example single service:

```bash
cd services/orchestrator && uv sync && PYTHONPATH=$(pwd)/../.. uv run pytest tests -v --tb=short
```

---

## API modes

- Request **`mode`**: `strict` or `hybrid` (forwarded per contract).
- Retrieval is grounded on the **ingested Qdrant corpus** unless you extend hybrid behavior; document any deployment-specific hybrid setup in `docs/architecture.md`.
