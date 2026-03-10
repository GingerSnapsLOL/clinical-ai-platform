"""Endpoint tests for llm-service: /health, /v1/generate."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
for p in (str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
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
    # Configure tokenizer mock: return fake input ids and decode calls.
    fake_input_ids = {"input_ids": [[1, 2, 3]]}

    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
    mock_tokenizer.__call__.return_value = fake_input_ids
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "mocked completion"

    # Configure model mock: generate extends the sequence with a few new tokens.
    mock_model = mock_model_cls.from_pretrained.return_value
    mock_model.device = "cpu"
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

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

