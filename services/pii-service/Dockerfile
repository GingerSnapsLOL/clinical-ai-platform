FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Include shared schemas and service code
COPY services ./services
COPY services/pii-service/pyproject.toml ./pyproject.toml
COPY services/pii-service/app ./app

RUN uv pip install --system --no-cache .

# PII service listens on 8020 (matches docker-compose and PII_SERVICE_URL)
ENV PII_SERVICE_PORT=8020

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]

