FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/ner-service/pyproject.toml ./pyproject.toml
COPY services/ner-service/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services ./services
COPY services/ner-service/app ./app
RUN uv sync --locked --no-editable
RUN uv pip install \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

ENV PATH="/app/.venv/bin:$PATH"
ENV NER_SERVICE_PORT=8030

EXPOSE 8030

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8030"]

