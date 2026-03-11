FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/scoring-service/pyproject.toml ./pyproject.toml
COPY services/scoring-service/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/scoring-service ./services/scoring-service
COPY services/scoring-service/app ./app
RUN uv sync --locked --no-editable

ENV PATH="/app/.venv/bin:$PATH"
ENV SCORING_SERVICE_PORT=8050

EXPOSE 8050

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8050"]

