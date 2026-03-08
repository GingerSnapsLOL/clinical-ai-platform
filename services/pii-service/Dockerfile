FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/pii-service/pyproject.toml ./pyproject.toml
COPY services/pii-service/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services ./services
COPY services/pii-service/app ./app
RUN uv sync --locked --no-editable

ENV PATH="/app/.venv/bin:$PATH"
ENV PII_SERVICE_PORT=8020

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]

