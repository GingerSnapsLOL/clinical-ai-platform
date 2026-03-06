FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Include shared schemas and service code
COPY services ./services
COPY services/orchestrator/pyproject.toml ./pyproject.toml
COPY services/orchestrator/app ./app

RUN pip install --no-cache-dir .

# Orchestrator listens on 8010 (matches docker-compose and ORCHESTRATOR_URL)
ENV ORCHESTRATOR_PORT=8010

EXPOSE 8010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]

