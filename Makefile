SHELL := /bin/bash

.PHONY: up down build rebuild lock logs ps health

# Generate uv.lock in each service for reproducible Docker builds
lock:
	@for svc in gateway-api orchestrator pii-service ner-service retrieval-service scoring-service; do \
		(cd services/$$svc && uv lock); \
	done

up:
	docker compose up -d

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

test:
	uv sync
	PYTHONPATH=. python -m pytest services/shared/tests -v --tb=short
	PYTHONPATH=.:services/gateway-api python -m pytest services/gateway-api/tests -v --tb=short
	PYTHONPATH=.:services/orchestrator python -m pytest services/orchestrator/tests -v --tb=short
	PYTHONPATH=.:services/pii-service python -m pytest services/pii-service/tests -v --tb=short
	PYTHONPATH=.:services/ner-service python -m pytest services/ner-service/tests -v --tb=short
	PYTHONPATH=.:services/retrieval-service python -m pytest services/retrieval-service/tests -v --tb=short
	PYTHONPATH=.:services/scoring-service python -m pytest services/scoring-service/tests -v --tb=short

