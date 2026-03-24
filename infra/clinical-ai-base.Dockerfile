FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pre-install large ML models that are shared across services
# so they are downloaded once into the base image.
RUN uv pip install --system \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl && \
    uv pip install --system \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# Service images will:
#   - reuse this base
#   - copy their own pyproject/uv.lock
#   - run `uv sync` to install their exact dependencies

