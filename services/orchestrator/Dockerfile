FROM clinical-ai-base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/orchestrator/pyproject.toml ./pyproject.toml
COPY services/orchestrator/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/orchestrator ./services/orchestrator
COPY services/orchestrator/app ./app
RUN uv sync --locked --no-editable

ENV PATH="/app/.venv/bin:$PATH"
ENV ORCHESTRATOR_PORT=8010

EXPOSE 8010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]

