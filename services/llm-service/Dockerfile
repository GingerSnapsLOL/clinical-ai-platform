FROM clinical-ai-base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/llm-service/pyproject.toml ./pyproject.toml
COPY services/llm-service/uv.lock ./uv.lock

RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/llm-service/app ./app

RUN uv sync --locked --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
ENV LLM_SERVICE_PORT=8060

EXPOSE 8060

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8060"]

