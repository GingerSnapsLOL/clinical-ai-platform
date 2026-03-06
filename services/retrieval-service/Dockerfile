FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Include shared schemas and service code
COPY services ./services
COPY services/retrieval-service/pyproject.toml ./pyproject.toml
COPY services/retrieval-service/app ./app

RUN pip install --no-cache-dir .

ENV RETRIEVAL_SERVICE_PORT=8040

EXPOSE 8040

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8040"]

