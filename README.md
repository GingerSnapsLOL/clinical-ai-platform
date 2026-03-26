## Clinical AI Platform

Privacy-first clinical AI backend focused on **PII redaction**, **biomedical NER**, **RAG over Qdrant with cross‑encoder reranking**, and **grounded answer synthesis**.  
All services are implemented as Dockerized FastAPI microservices with shared typed contracts and traceable HTTP calls.

This README is focused on **how to run the system today**.  
For architecture and long-term plans, see:

- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/api_contracts.md`

---

## Services and current behavior

### Service map

- **Gateway API (`:8000`)**
  - Public entrypoint for `POST /v1/ask`.
  - Validates incoming payload (`mode`, `note_text`, `question`, optional `trace_id`).
  - Ensures a `trace_id` is present and forwards request to orchestrator.

- **Orchestrator (`:8010`)**
  - Coordinates the full ask pipeline.
  - Calls PII redaction, NER extraction, retrieval, risk scoring, and answer generation services.
  - Aggregates downstream outputs into one response for gateway/frontends.
  - Design note:
    - I evaluated LangChain/LangGraph, but chose a custom orchestration layer because:
      - I needed full control over pipeline execution.
      - I wanted predictable latency and service boundaries.
      - I wanted simple debugging and observability.
      - I wanted to avoid deep framework lock-in.
    - For prototyping or quick agent demos, LangChain is useful, but for production systems I prefer explicit design.

- **PII Service (`:8020`)**
  - Endpoint: `POST /v1/redact`.
  - Detects sensitive fields and returns redacted text plus span-level details.

- **NER Service (`:8030`)**
  - Endpoint: `POST /v1/extract`.
  - Extracts biomedical/clinical entities from redacted text.

- **Retrieval Service (`:8040`)**
  - Endpoints: `POST /v1/ingest`, `POST /v1/retrieve`.
  - Ingests chunked clinical corpus into Qdrant embeddings.
  - Retrieves and reranks relevant passages for grounding.

- **Scoring Service (`:8050`)**
  - Endpoint: `POST /v1/score`.
  - Produces risk-oriented scoring output and feature contributions.
  - Current implementation is intentionally simple/stub-like.

- **LLM Service (`:8060`)**
  - Endpoint: `POST /v1/generate`.
  - Generates grounded answer text using retrieved evidence/context.

### End-to-end ask pipeline

`gateway-api` -> `orchestrator` -> `pii-service` -> `ner-service` -> `retrieval-service` -> `scoring-service` -> `llm-service` -> aggregated response

### Contracts and traceability

- Shared request/response schemas are centralized in `services/shared/schemas_v1.py`.
- Every service propagates `trace_id` for cross-service observability and debugging.
- `GET /health` is implemented across services for readiness and smoke checks.

---

## Local run instructions

### Prerequisites

- **Python**: 3.11+
- **uv**: for dependency management (optional if you only use Docker).
- **Docker + Docker Compose**: to run the full stack locally.
- **make**: to use the provided `Makefile` targets.
- **Hardware note**:
  - `llm-service` downloads and serves `Qwen/Qwen2.5-7B-Instruct`.  
  - On CPU-only machines, startup and inference will be slow; on GPUs with enough memory, `torch.cuda.is_available()` is used for float16.

#### How to install prerequisites (high level)

- **Python 3.11+**
  - Linux/macOS: use your package manager or `pyenv` (e.g. `pyenv install 3.11.8`).
  - Windows: use the official installer from `https://www.python.org` or the Microsoft Store.

- **uv**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  Or see the official instructions at `https://docs.astral.sh/uv/`.

- **Docker + Docker Compose**
  - Install Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux) from `https://docs.docker.com/get-docker/`.
  - Verify:

    ```bash
    docker --version
    docker compose version
    ```

- **make**
  - Linux: usually available via `build-essential`/`make` package (e.g. `sudo apt install make`).
  - macOS: install Xcode Command Line Tools (`xcode-select --install`).
  - Windows: use `make` from WSL (recommended) or install via a package manager like Chocolatey/MSYS2.

### 1. Environment and dependencies

- **Docker + Make (recommended)**
  - Copy environment:

    ```bash
    cp .env.example .env
    ```

  - Build locks, base image, all service images and run tests in one step:

    ```bash
    make all
    ```

  - Start or restart the stack:

    ```bash
    make up      # docker compose up -d
    make logs    # follow logs
    make down    # stop and remove volumes
    ```

- **Python dev setup (optional, for local uv runs)**

  ```bash
  uv sync
  ```

### 2. Ingest demo documents (once)

With the stack running and `retrieval-service` healthy:

```bash
python scripts/ingest_demo.py
```

This ingests `examples/clinical_docs*.json` into Qdrant via `retrieval-service` `/v1/ingest`.

### 3. Smoke-test the APIs

- **Health checks**

  ```bash
  curl http://localhost:8000/health
  curl http://localhost:8010/health
  curl http://localhost:8020/health
  curl http://localhost:8030/health
  curl http://localhost:8040/health
  curl http://localhost:8050/health
  curl http://localhost:8060/health
  ```

- **End-to-end `/v1/ask` (Strict mode)**

  ```bash
  curl -X POST http://localhost:8000/v1/ask \
    -H "Content-Type: application/json" \
    -d '{
      "mode": "strict",
      "note_text": "Patient John Doe has hypertension treated with lisinopril",
      "question": "What are the treatment risks?"
    }'
  ```

  Expected (assuming models and Qdrant are initialized correctly):
  - Non-empty `answer` grounded in retrieved passages.
  - `entities` contain at least hypertension and lisinopril.
  - `sources` contain up to 3 reranked passages.
  - `risk` present but derived from the stub scoring-service.

### 4. Helpful scripts

- `scripts/ingest_demo.py` – ingest demo corpus into Qdrant.
- `scripts/eval_retrieval.py` – quick retrieval sanity check:

  ```bash
  python scripts/eval_retrieval.py --query "hypertension treatment"
  ```

- `scripts/demo_m1.py` – basic `/v1/ask` demo (can load payload from `examples/ask_request.json`).
- `scripts/demo_m4.py` – richer `/v1/ask` demo that pretty‑prints answer, citations, sources, entities, and risk.

---

## Frontend Console (Next.js)

The repository includes a standalone frontend app in `frontend/` that acts as an internal clinical AI testing console for `/v1/ask`.

### Frontend stack

- Next.js App Router
- TypeScript
- ESLint
- Tailwind CSS

### Frontend structure

- `frontend/app/page.tsx` - landing page
- `frontend/app/ask/page.tsx` - main testing UI
- `frontend/app/api/ask/route.ts` - proxy route to backend gateway `/v1/ask`
- `frontend/components/*` - reusable UI panels and card components
- `frontend/lib/api.ts` - typed API client (`/api/ask`)
- `frontend/lib/types.ts` - typed request/response contracts used by UI

### Frontend request contract

Ask form sends:

- `mode` (`strict` or `hybrid`)
- `note_text`
- `question`

### Frontend features

- Portfolio-style dashboard layout with responsive two-column design
- Sticky request input column on desktop
- Demo prompt prefill buttons:
  - Hypertension treatment
  - Thiazide diuretics
  - Calcium channel blockers
  - Unknown query
- Answer panel with loading and error handling
- Grounding quality indicator near answers:
  - strong grounding (green)
  - weak grounding (yellow)
  - insufficient data (red)
- Sources panel redesigned as evidence cards:
  - title
  - relevance badge
  - metadata block
  - snippet
- Entities panel
- Risk assessment panel with colored status badge and structured explanation list
- Diagnostics panel:
  - total request time
  - retrieval time
  - llm time
- Compare mode (toggleable):
  - sends the same request twice
  - displays Answer A and Answer B side-by-side
  - shows latency differences and source differences
- Collapsible Trace/Debug panel:
  - `trace_id`
  - warnings
  - retrieval diagnostics
  - planner decisions (if available)

### Frontend environment

Create `frontend/.env.local`:

```env
BACKEND_BASE_URL=http://localhost:8000
```

### Run frontend locally

From repo root:

```bash
make frontend-install
make frontend-dev
```

`make up` also starts the frontend container now (via `docker-compose.yml`), so you can run the full stack with one command.

Quick verification:

```bash
docker compose ps
docker compose logs -f frontend
```

Or directly:

```bash
cd frontend
npm install
node ./node_modules/next/dist/bin/next dev
```

Open:

- `http://localhost:3000`
- `http://localhost:3000/ask`

---

## Repository structure (current)

```text
clinical-ai-platform/
  README.md
  docs/
    architecture.md   # current architecture (implemented)
    roadmap.md        # target architecture and future work
    api_contracts.md  # summary of shared API contracts
  services/
    gateway-api/
    orchestrator/
    pii-service/
    ner-service/
    retrieval-service/
    scoring-service/
    llm-service/
    shared/
  scripts/
  examples/
  infra/
  docker-compose.yml
  pyproject.toml
  uv.lock
```

---

## Environment variables (selection)

| Variable                | Description                            | Default                     |
|-------------------------|----------------------------------------|-----------------------------|
| `ORCHESTRATOR_URL`      | Orchestrator base URL (Docker)        | `http://orchestrator:8010` |
| `PII_SERVICE_URL`       | PII service base URL                  | `http://pii-service:8020`  |
| `NER_SERVICE_URL`       | NER service base URL                  | `http://ner-service:8030`  |
| `RETRIEVAL_SERVICE_URL` | Retrieval service base URL            | `http://retrieval-service:8040` |
| `SCORING_SERVICE_URL`   | Scoring service base URL              | `http://scoring-service:8050` |
| `LLM_BASE_URL`          | LLM service base URL (orchestrator)   | `http://llm-service:8060`  |
| `LLM_MODEL_NAME`        | HuggingFace model ID for llm-service  | `Qwen/Qwen2.5-7B-Instruct` |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant connection              | `qdrant` / `6333`          |
| `RETRIEVAL_URL`         | Retrieval URL used by scripts (local) | `http://localhost:8040`    |

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

Or per service, for example:

```bash
PYTHONPATH=.:services/pii-service uv run pytest services/pii-service/tests -v
PYTHONPATH=.:services/ner-service uv run pytest services/ner-service/tests -v
PYTHONPATH=.:services/retrieval-service uv run pytest services/retrieval-service/tests -v
PYTHONPATH=.:services/orchestrator uv run pytest services/orchestrator/tests -v
```

---

## Modes (runtime behavior)

- API accepts:
  - `strict`
  - `hybrid`

- In current local setup, retrieval is grounded on local indexed corpus (Qdrant + ingested documents).
- If you introduce external retrieval or web augmentation for hybrid flows, document deployment-specific behavior in `docs/architecture.md`.
