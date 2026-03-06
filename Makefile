SHELL := /bin/bash

.PHONY: up down build rebuild logs ps health

up:
	docker compose up -d

rebuild:
	docker compose up -d --build

build:
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

