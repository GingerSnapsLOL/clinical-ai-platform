# Clinical AI Platform — Implementation Plan

This document is a **concrete implementation plan** derived from the current codebase. For aspirational vision and long-term targets, see [`roadmap.md`](roadmap.md). For service topology, see [`architecture.md`](architecture.md). For API shapes, see [`api_contracts.md`](api_contracts.md).

**Last reviewed:** 2026-03-27 (align with repo state when editing).

---

## Executive snapshot

| Area | Status |
|------|--------|
| **End-to-end Strict RAG `/v1/ask`** | **Works:** gateway → orchestrator → PII → NER → retrieval ∥ scoring → relevance gate → LLM (or template fallback). |
| **Hybrid mode** | **Not implemented:** `Mode` includes `"hybrid"` in schemas; orchestrator has no web / external evidence path. |
| **Scoring** | **Stub:** fixed score/label; request entities/features not used for real inference. |
| **Retrieval** | **Real:** Qdrant + sentence-transformers embeddings + optional cross-encoder rerank; HTTP `/v1/ingest` and `scripts/ingest_qdrant.py` for `data/processed/datamix.jsonl`. |
| **LLM** | **Real:** `llm-service` supports transformers / vLLM via configuration. |
| **PII** | **Real:** Presidio with a focused operator set (e.g. person, phone, email). |
| **NER** | **Real:** SciSpaCy; normalized labels; confidence often unset. |
| **Postgres** | **In Compose:** not used by orchestrator for traces/audit in current code paths. |
| **Gateway security** | **Minimal:** CORS + forward; no authn/z or rate limiting in service code. |
| **CI (remote)** | **Missing in repo:** no `.github/workflows`; local `make test` / `make ci` exist. |
| **Schema drift** | Orchestrator builds **`citations`**; **`AskResponse`** in `services/shared/schemas_v1.py` has no `citations` field — Pydantic ignores extras on parse, so typed clients and OpenAPI may omit citations. |

---

## What already works

1. **Foundation** — Docker Compose (Postgres, Redis, Qdrant, microservices, frontend), shared HTTP client, timeouts, `X-Trace-Id`, structured logging, Makefile targets (`up`, `build`, `lock`, `test`, `ci`, `all-ingest`, etc.).

2. **Retrieval** — `/v1/retrieve`, `/v1/ingest`; offline pipeline via `scripts/download_*`, `parse_*`, `make_datamix.py`, `ingest_qdrant.py` into collection `clinical_docs`.

3. **Orchestration (strict)** — PII then NER; parallel retrieval and scoring; enriched query; Redis cache for retrieval and answers; relevance gate; LLM + fallback synthesis; latency fields on response (with semantic caveats below).

4. **Gateway** — `POST /v1/ask` with optional client `trace_id`, validation, proxy to orchestrator, structured errors.

5. **Frontend** — Next.js: `/ask` (diagnostics), `/chat` (minimal UI with context → `note_text`), `/api/ask` proxy.

6. **Evaluation (scripts)** — e.g. `scripts/eval_retrieval.py`, `scripts/eval_retrieval_datamix.py` (retrieval-focused; not a full golden-set product).

---

## What is stubbed or misleading

- **`scoring-service`:** static score/label/explanation; not driven by `ScoreRequest`.
- **Orchestrator:** initial risk may be stubbed until scoring returns; if scoring succeeds, risk is overwritten.
- **Hybrid:** no implementation; clients can send `mode: "hybrid"` without a defined behavior — should be documented or rejected explicitly.
- **Orchestrator docstring** may still describe the answer as “stub” though LLM + fallback exist — update for accuracy.
- **LLM timing UI:** `llm_time_ms` can read **0** when the LLM call is skipped (answer cache hit, retrieval gate failure, or exception before timing). Fallback synthesis time is not surfaced as “LLM” in the same field.
- **`docs/roadmap.md`** may still say “frontend planned” though a Next.js app exists — reconcile when editing roadmap.

---

## What is missing or weak

| Gap | Notes |
|-----|--------|
| **Real scoring + explanations** | Needed for trustworthy `risk` in clinical-adjacent UX. |
| **Contract alignment** | Add `citations` (and any diagnostics) to `AskResponse`; sync gateway, OpenAPI, frontend types. |
| **Persistence** | Use Postgres for audit traces, eval runs, or config when requirements firm up. |
| **Security & abuse** | Auth, rate limits, audit on gateway. |
| **Hybrid tools** | De-ID, allowlisted fetch, separate citation handling for web vs corpus. |
| **Evaluation as gate** | Golden sets, CI regression, optional LLM-judge (offline). |
| **Production deployment** | Beyond Compose: secrets, probes, backups, observability stack. |

---

## Phased roadmap

Priorities are ordered by **dependency** and **risk reduction**. Phases can overlap once Phase 1 is stable.

### Phase 1 — Foundation (highest priority)

**Goal:** Honest contracts, predictable API, minimal client surprise.

- Extend **`AskResponse`** with **`citations: List[CitationItem]`** (and any other orchestrator-only fields clients need); update **gateway** and **frontend** types.
- **`mode: "hybrid"`:** return **501** with a clear message, or implement behind a feature flag — avoid silent strict behavior.
- Refresh **orchestrator** docstrings and **docs** to match real LLM + fallback behavior.
- Document **latency fields** (cache hit, gate skip, fallback vs LLM) or add explicit flags / durations on the response.
- Add **CI** (e.g. GitHub Actions) running **`make ci`** on pull requests.

**Exit criteria:** OpenAPI matches runtime; citations visible to clients; hybrid behavior unambiguous; CI green on main.

---

### Phase 2 — Retrieval

**Goal:** Reliable corpus, measurable quality, operable index lifecycle.

- Align **env-config** for collection name, embedding model, reranker between **ingest scripts** and **retrieval-service**.
- Validate **metadata** surfaced in retrieve responses for citations and UI (title, url, source, section).
- **Operations:** health or admin checks for collection existence and approximate point count; Makefile or doc for reindex.
- Strengthen **eval:** pin a small eval snapshot; integrate `eval_retrieval_datamix.py` (or successor) into CI or scheduled runs.

**Exit criteria:** Documented ingest→retrieve path; eval fails on obvious regressions; runbook for empty/stale index.

---

### Phase 3 — Orchestration

**Goal:** Safer pipeline, clearer policies, path to hybrid.

- **Budgets:** token/latency caps for LLM calls where configurable.
- **Relevance gate:** tune thresholds; structured logging; optional product policy for “answer with low-confidence warning.”
- **Cache:** key versioning when models/collections change; TTL policy documented.
- **Hybrid (spike then productize):** de-identified queries only; allowlisted HTTP; tag passages as `source=web` vs corpus in prompts and citations.

**Exit criteria:** Per-request trace of major decisions; hybrid spike behind flag or explicit 501 until ready.

---

### Phase 4 — Scoring

**Goal:** Defensible risk or explicit abstention.

- Replace constant stub with **rules-based v1** or **trained model** using entities and optional structured features; real **`FeatureContribution`** where possible.
- If model is not ready: **`label`** and copy that reflect **uncertainty** instead of fake high scores.

**Exit criteria:** Tests with fixed inputs; no production path that always returns the same score.

---

### Phase 5 — Frontend

**Goal:** UX matches backend semantics.

- Show **warnings**, **risk**, and **timings** with labels (cached, skipped, fallback).
- Display **citations** when exposed by API.
- **Trace:** copy `trace_id`; later link to trace store.
- **E2E tests** (e.g. Playwright) against Compose or mocked backend.

**Exit criteria:** Happy path + error path covered; timings not misleading.

---

### Phase 6 — Evaluation

**Goal:** Regression safety for retrieval and answers.

- **Golden set:** note + question + expected citations or rubric.
- Metrics: retrieval hit@k, citation overlap; optional offline LLM judge.
- Store results in **CI artifacts** first; later **Postgres** or experiment tooling.

**Exit criteria:** Failing eval blocks merge or triggers alert.

---

### Phase 7 — Deployment

**Goal:** Environments beyond a single machine.

- **Secrets:** Docker secrets or cloud secret manager; no secrets baked into images.
- **Hardening:** non-root images, resource limits, readiness/liveness probes.
- **Backups:** Qdrant volumes + Postgres as needed.
- **Observability:** metrics and log aggregation (e.g. Prometheus/Grafana/Loki) as adopted by the team.

**Exit criteria:** Staged deploy documented; rollback and backup procedures exist.

---

## Recommended sequencing (first milestones)

1. **Phase 1** — Contracts, hybrid clarity, CI.  
2. **Phase 2** — Retrieval quality and eval hooks.  
3. **Phase 4** — Honest or real scoring (parallel with Phase 3 if staffed).  
4. **Phase 3** — Deeper orchestration and hybrid.  
5. **Phases 5–7** — In parallel where possible after Phase 1 types are stable.

---

## Maintenance

When making substantial architecture or contract changes, update this file in the same PR or immediately after, and keep **`api_contracts.md`** and **`architecture.md`** in sync.
