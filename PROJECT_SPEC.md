# Clinical AI Platform (Portfolio)

## Goal
Build a production-style, privacy-first Clinical AI system:
- PII redaction
- NER extraction
- RAG (vector DB) + optional reranking
- optional web evidence enrichment (Hybrid mode only)
- local LLM inference (vLLM/TGI)
- tabular risk scoring + SHAP explainability
- LangGraph orchestration with tool gating
- dockerized microservices, monitoring, evaluation, CI

## Modes
### Strict (default)
- No internet calls
- Only internal knowledge base (local corpus)

### Hybrid
- Web search and page fetch allowed
- Queries MUST be de-identified
- Allowlist domains only
- Web content is untrusted, sanitized, and cited

## Services
- frontend (Next.js)
- gateway-api (FastAPI): auth, audit, rate limits, single entrypoint
- orchestrator (FastAPI + LangGraph): deterministic pipeline, budgets, policies
- pii-service (FastAPI): PII detect + anonymize
- ner-service (FastAPI): medical entity extraction
- retrieval-service (FastAPI): ingest + retrieve from Qdrant + rerank
- scoring-service (FastAPI): risk scoring (CatBoost/XGB) + SHAP
- llm-service: vLLM/TGI inference server (OpenAI compatible)

Infrastructure:
- qdrant (vector DB)
- postgres (audit logs, traces)
- redis (cache/rate limiting)
Optional:
- mlflow + minio
- prometheus + grafana + loki

## Data
- NER: BC5CDR (open biomedical NER)
- RAG corpus: PubMed abstracts + open guidelines
- Tabular: UCI Heart/Diabetes (for scoring demo)
- Synthetic notes for PII demos

## Non-functional requirements
- Typed JSON contracts and versioning
- Trace IDs and structured logging
- Budgets: max tokens, max web calls, timeouts
- Evaluation suite for RAG/NER/scoring

## Milestones
M0: repo skeleton, docker-compose, /health everywhere
M1: orchestrator pipeline (stubbed), trace_id propagation
M2: retrieval (qdrant + embeddings) working
M3: vLLM serving integrated
M4: PII redaction + NER + scoring integrated
M5: Hybrid web enrichment + tool gating
M6: evaluation + dashboards + docs polish