## Clinical AI Platform

This repository contains a multi-service **Clinical AI Platform** prototype, designed around a LangGraph-based orchestrator and a set of focused ML microservices.

Milestone 0 (“Hello Platform”) is a minimal, stubbed implementation that brings up all core services and exposes a single end-to-end path:

- `frontend` → form for clinical note + question
- `gateway-api` → single public API `/v1/ask`
- `orchestrator` → stubbed state machine endpoint
- `pii-service`, `ner-service`, `retrieval-service`, `scoring-service` → FastAPI services with `/health` and stubbed endpoints

### Quick start

1. **Copy environment template**:

```bash
cp .env.example .env
```

2. **Build and run everything**:

```bash
docker compose up --build
```

3. **Verify health endpoints** (from your host):

- Gateway: `http://localhost:8000/health`
- Orchestrator: `http://localhost:8001/health`
- PII service: `http://localhost:8002/health`
- NER service: `http://localhost:8003/health`
- Retrieval service: `http://localhost:8004/health`
- Scoring service: `http://localhost:8005/health`
- Frontend: `http://localhost:3000/`

4. **Test the main API**:

Send a request to the gateway:

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "strict",
    "note_text": "55-year-old with hypertension...",
    "question": "What is the risk profile?",
    "user_context": {"lang": "en"}
  }'
```

You should receive a **stubbed JSON response** containing `answer`, `sources`, `entities`, `risk`, `trace_id`, and `pii_redacted`.

5. **Open the frontend**:

Navigate to `http://localhost:3000/` in your browser, enter a note and question, and submit. You should see the stubbed answer returned from the gateway.

### Project layout

The full architecture and contracts are documented in `PROJECT_SPEC.md`. The high-level layout:

- `services/` — all application services (gateway, orchestrator, PII, NER, retrieval, scoring, frontend)
- `infra/` — optional observability stack (Prometheus, Grafana, Loki)
- `data/` — knowledge base sources and scripts
- `eval/` — evaluation scaffolding for RAG, NER, and scoring

Later milestones will replace stubs with real ML logic, LangGraph orchestration, RAG over Qdrant, and risk scoring with explainability.

