FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY services/llm-service/pyproject.toml ./pyproject.toml
COPY services/llm-service/uv.lock ./uv.lock

RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/llm-service ./services/llm-service
COPY services/llm-service/app ./app

RUN uv sync --locked --no-editable

ENV LLM_SERVICE_PORT=8060

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8060"]

