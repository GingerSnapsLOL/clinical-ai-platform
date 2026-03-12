## Roadmap

This document captures the **target architecture and future work** for the Clinical AI Platform. Items in this file are intentionally aspirational and may not yet exist in the codebase.

For the currently implemented architecture, see `docs/architecture.md`. For concrete API contracts, see `docs/api_contracts.md`.

---

## Vision

Build a production-style, privacy-first Clinical AI system that combines:

- PII redaction on clinical text.
- Biomedical NER.
- RAG over a curated clinical/guideline corpus with optional reranking.
- Optional web evidence enrichment (Hybrid mode).
- Local LLM inference (vLLM/TGI or similar).
- Tabular risk scoring with SHAP-style explainability.
- Policy-aware orchestration (e.g. LangGraph) with budgets and tool gating.
- Dockerized microservices, monitoring, evaluation, and CI.

---

## Modes

### Strict (implemented, default)

- No internet calls.
- Uses only the local knowledge base (Qdrant + example corpus).

### Hybrid (planned)

Planned behavior (not yet implemented in code):

- Web search and page fetch as tools called from the orchestrator.
- Queries MUST be de-identified before leaving the platform.
- Allowlist of domains only; all fetched content treated as untrusted.
- Retrieved web content is sanitized and cited in responses.

---

## Target services (portfolio view)

The codebase already implements most of these as FastAPI services; some capabilities are still stubs or planned.

- **frontend (Next.js)** – planned
  - Web UI for clinicians and developers (answer exploration, evidence inspection, traces).
- **gateway-api (FastAPI)**
  - Auth, audit logging, and rate limiting (partially planned; current implementation is unauthenticated).
  - Single API entrypoint for clients.
- **orchestrator (FastAPI + planned LangGraph)**
  - Deterministic pipeline orchestration.
  - Budgeting (tokens, latency, external calls).
  - Policy enforcement and tool gating (e.g. Hybrid mode constraints).
- **pii-service (FastAPI)**
  - PII detection and anonymization for PHI/PII entities.
- **ner-service (FastAPI)**
  - Biomedical entity extraction over clinical text.
- **retrieval-service (FastAPI)**
  - Ingest and retrieve documents from Qdrant with optional reranking.
- **scoring-service (FastAPI)**
  - Risk scoring over tabular and structured features (CatBoost/XGB) with SHAP.
- **llm-service**
  - Local inference server (target: vLLM/TGI) with an OpenAI-compatible API.

### Infrastructure

Planned infrastructure components:

- **Vector DB**: Qdrant (already used).
- **Relational DB**: Postgres for audit logs, traces, and configuration.
- **Cache / rate limiting**: Redis.
- **Experiment tracking (optional)**: MLflow + Minio.
- **Monitoring and logging (optional)**: Prometheus + Grafana + Loki.

---

## Data sets (portfolio targets)

Planned examples and data sources:

- **NER**: BC5CDR (open biomedical NER).
- **RAG corpus**: PubMed abstracts and open-access guidelines (downsampled and curated).
- **Tabular**: UCI Heart/Diabetes datasets for toy risk scoring.
- **Text notes**: Synthetic notes for PII and NER demos.

Where licensing or size is an issue, the repo will rely on configuration + scripts to download or generate data rather than bundling everything.

---

## Non-functional requirements

The following NFRs guide future work:

- Typed JSON contracts and explicit versioning.
- Trace IDs and structured JSON logging across all services.
- Budgets for:
  - Max tokens.
  - Max web calls.
  - Request timeouts.
- Evaluation suites for:
  - RAG (retrieval quality).
  - NER (precision/recall on a held-out set).
  - Scoring (calibration, AUC).

---

## Milestones

These milestones describe the **intended evolution** of the platform. Some are already partially or fully implemented; others are future work.

- **M0 – Skeleton**
  - Repo skeleton and initial `docker-compose.yml`.
  - `/health` endpoints everywhere.

- **M1 – Orchestrator skeleton**
  - Orchestrator pipeline stubbed but wired (trace_id propagation).
  - Contracts defined in `services/shared/schemas_v1.py`.

- **M2 – Retrieval working**
  - Qdrant + embedding model integrated.
  - `/v1/ingest` and `/v1/retrieve` functional with basic reranking.

- **M3 – Local LLM serving**
  - Local LLM (initially `transformers`-hosted; later vLLM/TGI) responding on `/v1/generate`.
  - Orchestrator able to call LLM with grounded prompts.

- **M4 – Clinical pipeline integrated**
  - PII redaction + biomedical NER plugged into orchestrator.
  - Stub risk scoring returning structured risk blocks.
  - End-to-end `/v1/ask` delivering:
    - Answer text.
    - Entities.
    - Sources.
    - Risk block.

- **M5 – Hybrid web enrichment**
  - Orchestrator gains web tools for search/fetch under `Mode.hybrid`.
  - Tool gating and de-identification policies enforced.

- **M6 – Evaluation, monitoring, docs**
  - RAG/NER/scoring evaluation suites.
  - Basic Prometheus/Grafana dashboards.
  - Portfolio-ready documentation and examples.

---

## Current gap summary (vs. roadmap)

As of now:

- Implemented:
  - Backend microservices for gateway, orchestrator, PII, NER, retrieval, stub scoring, and LLM.
  - Local RAG over Qdrant with embeddings and cross-encoder reranking.
  - Strict mode behavior (no web) and local LLM-backed answers with deterministic fallback.
- Partially implemented:
  - Risk scoring (stub only).
  - Use of Postgres/Redis (wired in Compose but not used as primary storage for auth/audit/rate limiting).
- Not implemented yet:
  - Next.js frontend.
  - Hybrid web enrichment.
  - LangGraph-based orchestration and tool budgets.
  - Real tabular risk models with SHAP.
  - Monitoring, dashboards, and CI pipelines.

This separation keeps the repository README focused on **“how to run the system now”** while `docs/roadmap.md` and `docs/architecture.md` describe where the platform is headed.

## Roadmap

This document captures the **target architecture and future work** for the Clinical AI Platform. Items in this file are intentionally aspirational and may not yet exist in the codebase.

For the currently implemented architecture, see `docs/architecture.md`. For concrete API contracts, see `docs/api_contracts.md`.

---

## Vision

Build a production-style, privacy-first Clinical AI system that combines:

- PII redaction on clinical text.
- Biomedical NER.
- RAG over a curated clinical/guideline corpus with optional reranking.
- Optional web evidence enrichment (Hybrid mode).
- Local LLM inference (vLLM/TGI or similar).
- Tabular risk scoring with SHAP-style explainability.
- Policy-aware orchestration (e.g. LangGraph) with budgets and tool gating.
- Dockerized microservices, monitoring, evaluation, and CI.

---

## Modes

### Strict (implemented, default)

- No internet calls.
- Uses only the local knowledge base (Qdrant + example corpus).

### Hybrid (planned)

Planned behavior (not yet implemented in code):

- Web search and page fetch as tools called from the orchestrator.
- Queries MUST be de-identified before leaving the platform.
- Allowlist of domains only; all fetched content treated as untrusted.
- Retrieved web content is sanitized and cited in responses.

---

## Target services (portfolio view)

The codebase already implements most of these as FastAPI services; some capabilities are still stubs or planned.

- **frontend (Next.js)** – planned
  - Web UI for clinicians and developers (answer exploration, evidence inspection, traces).
- **gateway-api (FastAPI)**
  - Auth, audit logging, and rate limiting (partially planned; current implementation is unauthenticated).
  - Single API entrypoint for clients.
- **orchestrator (FastAPI + planned LangGraph)**
  - Deterministic pipeline orchestration.
  - Budgeting (tokens, latency, external calls).
  - Policy enforcement and tool gating (e.g. Hybrid mode constraints).
- **pii-service (FastAPI)**
  - PII detection and anonymization for PHI/PII entities.
- **ner-service (FastAPI)**
  - Biomedical entity extraction over clinical text.
- **retrieval-service (FastAPI)**
  - Ingest and retrieve documents from Qdrant with optional reranking.
- **scoring-service (FastAPI)**
  - Risk scoring over tabular and structured features (CatBoost/XGB) with SHAP.
- **llm-service**
  - Local inference server (target: vLLM/TGI) with an OpenAI-compatible API.

### Infrastructure

Planned infrastructure components:

- **Vector DB**: Qdrant (already used).
- **Relational DB**: Postgres for audit logs, traces, and configuration.
- **Cache / rate limiting**: Redis.
- **Experiment tracking (optional)**: MLflow + Minio.
- **Monitoring and logging (optional)**: Prometheus + Grafana + Loki.

---

## Data sets (portfolio targets)

Planned examples and data sources:

- **NER**: BC5CDR (open biomedical NER).
- **RAG corpus**: PubMed abstracts and open-access guidelines (downsampled and curated).
- **Tabular**: UCI Heart/Diabetes datasets for toy risk scoring.
- **Text notes**: Synthetic notes for PII and NER demos.

Where licensing or size is an issue, the repo will rely on configuration + scripts to download or generate data rather than bundling everything.

---

## Non-functional requirements

The following NFRs guide future work:

- Typed JSON contracts and explicit versioning.
- Trace IDs and structured JSON logging across all services.
- Budgets for:
  - Max tokens.
  - Max web calls.
  - Request timeouts.
- Evaluation suites for:
  - RAG (retrieval quality).
  - NER (precision/recall on a held-out set).
  - Scoring (calibration, AUC).

---

## Milestones

These milestones describe the **intended evolution** of the platform. Some are already partially or fully implemented; others are future work.

- **M0 – Skeleton**
  - Repo skeleton and initial `docker-compose.yml`.
  - `/health` endpoints everywhere.

- **M1 – Orchestrator skeleton**
  - Orchestrator pipeline stubbed but wired (trace_id propagation).
  - Contracts defined in `services/shared/schemas_v1.py`.

- **M2 – Retrieval working**
  - Qdrant + embedding model integrated.
  - `/v1/ingest` and `/v1/retrieve` functional with basic reranking.

- **M3 – Local LLM serving**
  - Local LLM (initially `transformers`-hosted; later vLLM/TGI) responding on `/v1/generate`.
  - Orchestrator able to call LLM with grounded prompts.

- **M4 – Clinical pipeline integrated**
  - PII redaction + biomedical NER plugged into orchestrator.
  - Stub risk scoring returning structured risk blocks.
  - End-to-end `/v1/ask` delivering:
    - Answer text.
    - Entities.
    - Sources.
    - Risk block.

- **M5 – Hybrid web enrichment**
  - Orchestrator gains web tools for search/fetch under `Mode.hybrid`.
  - Tool gating and de-identification policies enforced.

- **M6 – Evaluation, monitoring, docs**
  - RAG/NER/scoring evaluation suites.
  - Basic Prometheus/Grafana dashboards.
  - Portfolio-ready documentation and examples.

---

## Current gap summary (vs. roadmap)

As of now:

- Implemented:
  - Backend microservices for gateway, orchestrator, PII, NER, retrieval, stub scoring, and LLM.
  - Local RAG over Qdrant with embeddings and cross-encoder reranking.
  - Strict mode behavior (no web) and local LLM-backed answers with deterministic fallback.
- Partially implemented:
  - Risk scoring (stub only).
  - Use of Postgres/Redis (wired in Compose but not used as primary storage for auth/audit/rate limiting).
- Not implemented yet:
  - Next.js frontend.
  - Hybrid web enrichment.
  - LangGraph-based orchestration and tool budgets.
  - Real tabular risk models with SHAP.
  - Monitoring, dashboards, and CI pipelines.

This separation keeps the repository README focused on **“how to run the system now”** while `docs/roadmap.md` and `docs/architecture.md` describe where the platform is headed.

