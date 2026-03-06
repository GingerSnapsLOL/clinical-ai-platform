FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Include shared schemas and service code
COPY services ./services
COPY services/ner-service/pyproject.toml ./pyproject.toml
COPY services/ner-service/app ./app

RUN pip install --no-cache-dir .

ENV NER_SERVICE_PORT=8030

EXPOSE 8030

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8030"]

