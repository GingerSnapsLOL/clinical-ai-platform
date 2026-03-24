FROM clinical-ai-base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY services/gateway-api/pyproject.toml ./pyproject.toml
COPY services/gateway-api/uv.lock ./uv.lock
RUN uv sync --locked --no-install-project

COPY services/shared ./services/shared
COPY services/gateway-api/app ./app
RUN uv sync --locked --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
ENV GATEWAY_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


