FROM clinical-ai-base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/ner-service/pyproject.toml ./pyproject.toml
COPY services/ner-service/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/ner-service ./services/ner-service
COPY services/ner-service/app ./app
RUN uv sync --locked --no-editable

ENV PATH="/app/.venv/bin:$PATH"
ENV NER_SERVICE_PORT=8030

EXPOSE 8030

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8030"]

