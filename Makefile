SHELL := /bin/bash

.PHONY: up down build rebuild lock logs ps health reset

# Generate uv.lock in each service for reproducible Docker builds
lock:
	@for svc in gateway-api orchestrator pii-service ner-service retrieval-service scoring-service llm-service; do \
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
	@for svc in gateway-api orchestrator pii-service ner-service retrieval-service scoring-service llm-service; do \
		(cd services/$$svc && uv sync && PYTHONPATH=$$(pwd)/../.. uv run pytest tests -v --tb=short); \
	done

