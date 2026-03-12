## Architecture

This document describes the **current** architecture of the Clinical AI Platform as implemented in this repository.

### High-level overview

- **Pattern**: Dockerized FastAPI microservices with shared typed contracts and structured logging.
- **Domain focus**:
  - PII redaction for clinical notes.
  - Biomedical named entity recognition (NER).
  - Retrieval-augmented generation (RAG) over a local guideline-like corpus.
  - Local LLM-based answer generation with deterministic fallback.
- **Data flow** (implemented):
  1. Client calls `gateway-api` `/v1/ask`.
  2. `gateway-api` validates the request, creates/propagates `trace_id`, and forwards to `orchestrator`.
  3. `orchestrator` calls:
     - `pii-service` for PII redaction.
     - `ner-service` for biomedical entity extraction.
     - `retrieval-service` for Qdrant-based retrieval + optional reranking.
     - `scoring-service` for **stubbed** risk scoring.
     - `llm-service` for grounded answer generation, with a deterministic fallback when LLM is unavailable.
  4. `orchestrator` assembles a structured `AskResponse` and returns it to `gateway-api`, which returns it to the client.

### Services

- **gateway-api (FastAPI)**
  - Public HTTP entrypoint.
  - Exposes:
    - `GET /health`
    - `POST /v1/ask`
  - Responsibilities:
    - Validate incoming requests (`AskRequestIn`).
    - Generate `trace_id` when missing and propagate it downstream.
    - Call `orchestrator` `/v1/ask` using the shared HTTP client.
    - Map network/HTTP failures into a structured `ErrorInfo` payload.

- **orchestrator (FastAPI)**
  - Internal orchestration service implementing the end-to-end pipeline.
  - Exposes:
    - `GET /health`
    - `POST /v1/ask`
  - Responsibilities:
    - Call PII, NER, retrieval, scoring, and LLM services in order using `httpx` via `services.shared.http_client`.
    - Build an enriched retrieval query (question + entity hints + lightweight note summary).
    - Convert retrieval results into `SourceItem` objects.
    - Call `llm-service` with a grounded prompt built from the question, entities, sources, and risk block.
    - On LLM failure, fall back to deterministic synthesis using `_synthesize_answer`.
    - Return a typed `AskResponse`.
  - **Note**: The current orchestrator is implemented as straightforward FastAPI handlers plus helper functions (no LangGraph in code).

- **pii-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/redact`
  - Responsibilities:
    - Initialize Presidio + spaCy model on startup.
    - Detect PII entities (PERSON, PHONE_NUMBER, EMAIL_ADDRESS).
    - Anonymize text with replacements `[PERSON]`, `[PHONE]`, `[EMAIL]`.
    - Return `RedactResponse` with `redacted_text` and `PIISpan` items.

- **ner-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/extract`
  - Responsibilities:
    - Initialize SciSpaCy NER model on startup.
    - Run biomedical NER on request text.
    - Normalize labels (e.g. DRUG → CHEMICAL) into a small, consistent set.
    - Return `ExtractResponse` with typed `EntityItem` list.

- **retrieval-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/retrieve`
    - `POST /v1/ingest`
  - Responsibilities:
    - On startup:
      - Load `SentenceTransformer("all-MiniLM-L6-v2")` for embeddings.
      - Load `CrossEncoder("ms-marco-MiniLM-L-6-v2")` for reranking.
      - Initialize a `clinical_docs` collection in Qdrant.
    - `/v1/ingest`:
      - Chunk documents into ~300–500 token segments.
      - Embed chunks and upsert into Qdrant with stable IDs.
    - `/v1/retrieve`:
      - Embed query.
      - Search Qdrant.
      - Deduplicate passages by text.
      - Optionally rerank with cross-encoder.
      - Limit to top 3 passages when rerank is enabled.

- **scoring-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/score`
  - Responsibilities:
    - Currently implemented as a **stub**:
      - Returns a fixed score (0.72), label `"high"`, and a small list of `FeatureContribution` values.
    - Provides a realistic contract for future risk-scoring models while keeping implementation simple.

- **llm-service (FastAPI + transformers)**
  - Exposes:
    - `GET /health`
    - `POST /v1/generate`
  - Responsibilities:
    - Load `Qwen/Qwen2.5-7B-Instruct` on startup using `transformers`.
    - Accept prompts plus generation parameters.
    - Run `model.generate` and return decoded text plus usage counts.
  - **Note**: Current implementation is a custom FastAPI wrapper over a `transformers` model (not vLLM/TGI).

### Shared layer (`services/shared`)

The `services/shared` package is intentionally restricted to **contracts and cross-cutting concerns**, not business logic:

- `schemas_v1.py`:
  - Pydantic v2 models for:
    - Health, error, and status envelopes.
    - Gateway/orchestrator `AskRequest` / `AskResponse`.
    - PII, NER, retrieval, scoring, and LLM request/response payloads.
  - Enum-like `Mode` and `Status` literals.
- `http_client.py`:
  - Typed `post_typed` helper for inter-service JSON calls using `httpx.AsyncClient`.
  - Timeout configuration and `X-Trace-Id` header propagation.
- `logging_util.py`:
  - Context-based `trace_id` storage.
  - JSON log formatter and FastAPI middleware factory for structured logs.
- `llm_client.py`:
  - Typed `LLMClient` abstraction for orchestrator → llm-service calls.

No business or domain-specific scoring, retrieval, NER, or PII logic lives in `services/shared`; that logic resides only within each service’s own `app/` package.

### Infrastructure

- **docker-compose.yml**
  - Orchestrates:
    - All Python services (gateway, orchestrator, pii, ner, retrieval, scoring, llm).
    - Qdrant.
    - Postgres and Redis containers (wired but not yet used extensively by the services).
- **Prompts**
  - `prompts/grounded_answer_synthesis.txt` provides the template for grounded answer generation used by orchestrator + llm-service.

For future/target architecture and planned evolutions, see `docs/roadmap.md`.

## Architecture

This document describes the **current** architecture of the Clinical AI Platform as implemented in this repository.

### High-level overview

- **Pattern**: Dockerized FastAPI microservices with shared typed contracts and structured logging.
- **Domain focus**:
  - PII redaction for clinical notes.
  - Biomedical named entity recognition (NER).
  - Retrieval-augmented generation (RAG) over a local guideline-like corpus.
  - Local LLM-based answer generation with deterministic fallback.
- **Data flow** (implemented):
  1. Client calls `gateway-api` `/v1/ask`.
  2. `gateway-api` validates the request, creates/propagates `trace_id`, and forwards to `orchestrator`.
  3. `orchestrator` calls:
     - `pii-service` for PII redaction.
     - `ner-service` for biomedical entity extraction.
     - `retrieval-service` for Qdrant-based retrieval + optional reranking.
     - `scoring-service` for **stubbed** risk scoring.
     - `llm-service` for grounded answer generation, with a deterministic fallback when LLM is unavailable.
  4. `orchestrator` assembles a structured `AskResponse` and returns it to `gateway-api`, which returns it to the client.

### Services

- **gateway-api (FastAPI)**
  - Public HTTP entrypoint.
  - Exposes:
    - `GET /health`
    - `POST /v1/ask`
  - Responsibilities:
    - Validate incoming requests (`AskRequestIn`).
    - Generate `trace_id` when missing and propagate it downstream.
    - Call `orchestrator` `/v1/ask` using the shared HTTP client.
    - Map network/HTTP failures into a structured `ErrorInfo` payload.

- **orchestrator (FastAPI)**
  - Internal orchestration service implementing the end-to-end pipeline.
  - Exposes:
    - `GET /health`
    - `POST /v1/ask`
  - Responsibilities:
    - Call PII, NER, retrieval, scoring, and LLM services in order using `httpx` via `services.shared.http_client`.
    - Build an enriched retrieval query (question + entity hints + lightweight note summary).
    - Convert retrieval results into `SourceItem` objects.
    - Call `llm-service` with a grounded prompt built from the question, entities, sources, and risk block.
    - On LLM failure, fall back to deterministic synthesis using `_synthesize_answer`.
    - Return a typed `AskResponse`.
  - **Note**: Although the project spec mentions LangGraph and tool gating, the current orchestrator is implemented as straightforward FastAPI handlers plus helper functions (no LangGraph in code).

- **pii-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/redact`
  - Responsibilities:
    - Initialize Presidio + spaCy model on startup.
    - Detect PII entities (PERSON, PHONE_NUMBER, EMAIL_ADDRESS).
    - Anonymize text with replacements `[PERSON]`, `[PHONE]`, `[EMAIL]`.
    - Return `RedactResponse` with `redacted_text` and `PIISpan` items.

- **ner-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/extract`
  - Responsibilities:
    - Initialize SciSpaCy NER model on startup.
    - Run biomedical NER on request text.
    - Normalize labels (e.g. DRUG → CHEMICAL) into a small, consistent set.
    - Return `ExtractResponse` with typed `EntityItem` list.

- **retrieval-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/retrieve`
    - `POST /v1/ingest`
  - Responsibilities:
    - On startup:
      - Load `SentenceTransformer("all-MiniLM-L6-v2")` for embeddings.
      - Load `CrossEncoder("ms-marco-MiniLM-L-6-v2")` for reranking.
      - Initialize a `clinical_docs` collection in Qdrant.
    - `/v1/ingest`:
      - Chunk documents into ~300–500 token segments.
      - Embed chunks and upsert into Qdrant with stable IDs.
    - `/v1/retrieve`:
      - Embed query.
      - Search Qdrant.
      - Deduplicate passages by text.
      - Optionally rerank with cross-encoder.
      - Limit to top 3 passages when rerank is enabled.

- **scoring-service (FastAPI)**
  - Exposes:
    - `GET /health`
    - `POST /v1/score`
  - Responsibilities:
    - Currently implemented as a **stub**:
      - Returns a fixed score (0.72), label `"high"`, and a small list of `FeatureContribution` values.
    - Provides a realistic contract for future risk-scoring models while keeping implementation simple.

- **llm-service (FastAPI + transformers)**
  - Exposes:
    - `GET /health`
    - `POST /v1/generate`
  - Responsibilities:
    - Load `Qwen/Qwen2.5-7B-Instruct` on startup using `transformers`.
    - Accept prompts plus generation parameters.
    - Run `model.generate` and return decoded text plus usage counts.
  - **Note**: The project spec mentions vLLM/TGI and an OpenAI-compatible API; current implementation is a custom FastAPI wrapper over a `transformers` model.

### Shared layer (`services/shared`)

The `services/shared` package is intentionally restricted to **contracts and cross-cutting concerns**, not business logic:

- `schemas_v1.py`:
  - Pydantic v2 models for:
    - Health, error, and status envelopes.
    - Gateway/orchestrator `AskRequest` / `AskResponse`.
    - PII, NER, retrieval, scoring, and LLM request/response payloads.
  - Enum-like `Mode` and `Status` literals.
- `http_client.py`:
  - Typed `post_typed` helper for inter-service JSON calls using `httpx.AsyncClient`.
  - Timeout configuration and `X-Trace-Id` header propagation.
- `logging_util.py`:
  - Context-based `trace_id` storage.
  - JSON log formatter and FastAPI middleware factory for structured logs.
- `llm_client.py`:
  - Typed `LLMClient` abstraction for orchestrator → llm-service calls.

No business or domain-specific scoring, retrieval, NER, or PII logic lives in `services/shared`; that logic resides only within each service’s own `app/` package.

### Infrastructure

- **docker-compose.yml**
  - Orchestrates:
    - All Python services (gateway, orchestrator, pii, ner, retrieval, scoring, llm).
    - Qdrant.
    - Postgres and Redis containers (wired but not yet used extensively by the services).
- **Prompts**
  - `prompts/grounded_answer_synthesis.txt` provides the template for grounded answer generation used by orchestrator + llm-service.

For future/target architecture and planned evolutions, see `docs/roadmap.md`.

