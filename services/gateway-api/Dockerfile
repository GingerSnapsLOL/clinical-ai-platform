FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Include shared schemas and service code
COPY services ./services
COPY services/gateway-api/pyproject.toml ./pyproject.toml
COPY services/gateway-api/app ./app

RUN pip install --no-cache-dir .

ENV GATEWAY_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


