## Clinical AI Platform

Privacy-first clinical AI backend focused on **PII redaction**, **biomedical NER**, **RAG over Qdrant with cross‑encoder reranking**, and **grounded answer synthesis**.  
All services are implemented as Dockerized FastAPI microservices with shared typed contracts and traceable HTTP calls.

This README is focused on **how to run the system today**.  
For architecture and long-term plans, see:

- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/api_contracts.md`

---

## What works today

- **Health checks**
  - Each service exposes `/health` returning a typed `HealthResponse`.
  - Docker Compose wires all services plus infra containers (Qdrant, Postgres, Redis, LLM).

- **Core APIs**
  - **Gateway `/v1/ask`**
    - Validates public input (`note_text`, `question`, optional `mode`, optional `trace_id`).
    - Generates a UUID trace ID when not provided and forwards to orchestrator.
  - **Orchestrator `/v1/ask`**
    - Calls:
      - `pii-service` `/v1/redact` on the original note text.
      - `ner-service` `/v1/extract` on the redacted text.
      - `retrieval-service` `/v1/retrieve` with an enriched query (question + entities + lightweight note summary).
      - `scoring-service` `/v1/score` using extracted entities (currently stub scoring).
      - `llm-service` `/v1/generate` with a grounded prompt built from entities, passages, and risk.
    - On any LLM error, falls back to a deterministic, template-based answer that still uses retrieved passages and the risk block.
    - Returns `AskResponse` including `answer`, `entities`, `sources`, `citations`, `risk`, `trace_id`, and `pii_redacted`.
  - **PII `/v1/redact`**
    - Detects and anonymizes PII, returning `redacted_text` and structured spans with confidence scores.
  - **NER `/v1/extract`**
    - Runs SciSpaCy NER and returns normalized `EntityItem` objects (type, text, span).
  - **Retrieval `/v1/retrieve` and `/v1/ingest`**
    - Ingest: chunks documents (≈300–500 tokens), embeds with `all-MiniLM-L6-v2`, and upserts into Qdrant.
    - Retrieve: embeds query, searches Qdrant, deduplicates passages, optionally reranks with `ms-marco-MiniLM-L-6-v2`, and returns up to 3 top passages when rerank is enabled.
  - **Scoring `/v1/score`**
    - Returns a fixed high‑risk score and a small list of feature contributions; intended as a stub for now.
  - **LLM `/v1/generate`**
    - Wraps a locally hosted Qwen model and returns generated text plus simple token usage metrics.

- **Traceability and shared contracts**
  - All services use a shared `trace_id` for structured logging.
  - Request/response schemas are centralized in `services/shared/schemas_v1.py`, with tests to keep them in sync.

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

- **Strict (implemented)**  
  - No internet calls; only local corpus (Qdrant + examples).  
  - This is the **effective behavior for all current requests**, regardless of the `mode` field.

- **Hybrid (not implemented yet)**  
  - Planned: web search/page fetch with strict de‑identification and allowlisted domains.  
  - As of now, setting `mode="hybrid"` has no effect; all requests are processed in strict mode.
