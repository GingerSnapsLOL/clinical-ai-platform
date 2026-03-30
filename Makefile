SHELL := /bin/bash

.PHONY: setup up init down build rebuild lock logs ps health reset test ingest all-ingest base-image all ci frontend-install frontend-dev frontend

SERVICES := gateway-api orchestrator pii-service ner-service retrieval-service scoring-service llm-service

# Generate uv.lock in each service for reproducible Docker builds
lock:
	@for svc in $(SERVICES); do \
		(cd services/$$svc && uv lock); \
	done

base-image:
	docker build -f infra/clinical-ai-base.Dockerfile -t clinical-ai-base .

setup:
	@if [ ! -f .env ]; then cp .env.example .env; fi
	uv sync

up: setup lock base-image build
	docker compose up -d
	@if [ "$(INIT)" = "1" ]; then $(MAKE) ingest; fi

init: ingest

rebuild: lock
	docker compose up -d --build

build: lock
	docker compose build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

health:
	@curl -fsS http://localhost:8000/health >/dev/null && echo "gateway-api OK"
	@curl -fsS http://localhost:8010/health >/dev/null && echo "orchestrator OK"
	@curl -fsS http://localhost:8020/health >/dev/null && echo "pii-service OK"
	@curl -fsS http://localhost:8030/health >/dev/null && echo "ner-service OK"
	@curl -fsS http://localhost:8040/health >/dev/null && echo "retrieval-service OK"
	@curl -fsS http://localhost:8050/health >/dev/null && echo "scoring-service OK"
	@curl -fsS http://localhost:8060/health >/dev/null && echo "llm-service OK"
	echo "All services OK"

# Full reset: stop, no-cache rebuild, and start everything
reset: down
	docker compose build --no-cache
	docker compose up -d

test:
	uv sync
	PYTHONPATH=. uv run pytest services/shared/tests -v --tb=short
	@for svc in $(SERVICES); do \
		(cd services/$$svc && uv sync && PYTHONPATH=$$(pwd)/../.. uv run pytest tests -v --tb=short); \
	done

# Full retrieval index: download sources, parse to JSONL, merge datamix, embed into Qdrant.
# Requires Qdrant reachable at http://localhost:6333 (e.g. docker compose up qdrant).
# DailyMed/MedlinePlus downloads are large and slow; optional interim file: data/interim/synthetic.jsonl
ingest: setup
	set -euo pipefail; \
	uv run python scripts/download_medlineplus.py; \
	uv run python scripts/download_dailymed.py; \
	uv run python scripts/parse_medlineplus.py; \
	uv run python scripts/parse_dailymed.py; \
	uv run python scripts/make_datamix.py; \
	uv run python scripts/ingest_qdrant.py

# Docker-first build path (no local tests)
all: lock base-image build

# setup + lock + base-image + build + docker compose up -d, then full ingest (datamix → Qdrant).
all-ingest: up
	$(MAKE) ingest

# CI path (includes local tests)
ci: all test

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend: frontend-install frontend-dev

