"""Endpoint tests for llm-service: /health, /v1/generate.

Run from service dir (see repo Makefile):

  cd services/llm-service
  PYTHONPATH=$(pwd)/../.. uv run pytest tests -v

Or from repo root:

  make test
"""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
for p in (str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _configure_hf_mocks(mock_model_cls, mock_tokenizer_cls) -> None:
    """Minimal HF stubs so FastAPI startup can finish without downloading a model."""
    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value

    # generate() does: tokenizer(...).to(model.device); avoid mock_tokenizer.__call__
    # (Python 3.12 + MagicMock: __call__ is a real method, not a configurable child mock).
    enc = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    class _Batch:
        def to(self, _device):
            return enc

    mock_tokenizer.return_value = _Batch()
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "mocked completion"

    mock_model = mock_model_cls.from_pretrained.return_value
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


def test_route_table_includes_health_and_generate():
    """Confirm routes are registered on the FastAPI app (no HTTP, no startup side effects)."""
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    by_path = {r.path: r for r in api_routes}
    assert "/health" in by_path
    assert "GET" in by_path["/health"].methods
    assert "/v1/generate" in by_path
    assert "POST" in by_path["/v1/generate"].methods


def test_openapi_schema_includes_health_and_generate():
    """Confirm OpenAPI documents the same paths and methods."""
    schema = app.openapi()
    assert "/health" in schema["paths"]
    assert "get" in schema["paths"]["/health"]
    assert "/v1/generate" in schema["paths"]
    assert "post" in schema["paths"]["/v1/generate"]


@patch("app.main.AutoTokenizer")
@patch("app.main.AutoModelForCausalLM")
def test_health(mock_model_cls, mock_tokenizer_cls):
    """Health must stay lightweight: mock HF so this passes when run alone."""
    _configure_hf_mocks(mock_model_cls, mock_tokenizer_cls)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "llm-service"


@patch("app.main.AutoTokenizer")
@patch("app.main.AutoModelForCausalLM")
def test_generate_contract_with_mock_model(mock_model_cls, mock_tokenizer_cls):
    """
    Contract test for /v1/generate that does not require loading a real 7B model.

    We mock the HF components so that startup completes quickly and generation is cheap.
    """
    _configure_hf_mocks(mock_model_cls, mock_tokenizer_cls)

    r = client.post(
        "/v1/generate",
        json={
            "trace_id": "tid-llm-1",
            "prompt": "Test prompt",
            "max_tokens": 16,
            "temperature": 0.2,
        },
    )

    assert r.status_code == 200
    data = r.json()

    assert data["trace_id"] == "tid-llm-1"
    assert isinstance(data.get("text"), str)
    assert data["text"] != ""
    usage = data.get("usage", {})
    assert isinstance(usage, dict)
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage

