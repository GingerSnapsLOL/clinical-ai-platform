FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Include shared schemas and service code
COPY services ./services
COPY services/scoring-service/pyproject.toml ./pyproject.toml
COPY services/scoring-service/app ./app

RUN pip install --no-cache-dir .

ENV SCORING_SERVICE_PORT=8050

EXPOSE 8050

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8050"]

