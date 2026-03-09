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

# Pre-install spaCy model so Presidio does not try to download via pip at runtime (venv has no pip).
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

ENV PATH="/app/.venv/bin:$PATH"
ENV PII_SERVICE_PORT=8020

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]

