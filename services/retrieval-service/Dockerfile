FROM clinical-ai-base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/retrieval-service/pyproject.toml ./pyproject.toml
COPY services/retrieval-service/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/retrieval-service/app ./app
RUN uv sync --locked --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
ENV RETRIEVAL_SERVICE_PORT=8040

EXPOSE 8040

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8040"]

