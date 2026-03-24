#!/usr/bin/env python3
"""Probe public HTTP endpoints for the local Clinical AI Platform stack.

By default checks GET /health on each FastAPI service (ports from docker-compose).
With --smoke, also POSTs a minimal /v1/ask through the gateway.

Usage (from repo root, with stack running):

  uv run python scripts/check_endpoints.py
  uv run python scripts/check_endpoints.py --smoke

Override base URLs with env vars if needed (same host, different ports):

  GATEWAY_URL, ORCHESTRATOR_URL, PII_SERVICE_URL, NER_SERVICE_URL,
  RETRIEVAL_SERVICE_URL, SCORING_SERVICE_URL, LLM_SERVICE_URL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULTS = {
    "gateway-api": ("GATEWAY_URL", "http://localhost:8000"),
    "orchestrator": ("ORCHESTRATOR_URL", "http://localhost:8010"),
    "pii-service": ("PII_SERVICE_URL", "http://localhost:8020"),
    "ner-service": ("NER_SERVICE_URL", "http://localhost:8030"),
    "retrieval-service": ("RETRIEVAL_SERVICE_URL", "http://localhost:8040"),
    "scoring-service": ("SCORING_SERVICE_URL", "http://localhost:8050"),
    "llm-service": ("LLM_SERVICE_URL", "http://localhost:8060"),
}


@dataclass
class CheckResult:
    name: str
    url: str
    ok: bool
    detail: str
    status_code: int | None = None
    body: Any = None


def _service_base(env_key: str, default_url: str) -> str:
    return os.getenv(env_key, default_url).rstrip("/")


def check_health(client: httpx.Client, name: str, base_url: str) -> CheckResult:
    url = f"{base_url}/health"
    try:
        r = client.get(url, timeout=10.0)
        if r.status_code != 200:
            return CheckResult(name, url, False, f"HTTP {r.status_code}", r.status_code, None)
        try:
            data = r.json()
        except json.JSONDecodeError:
            return CheckResult(name, url, False, "not JSON", r.status_code, r.text[:200])
        if data.get("status") != "ok":
            return CheckResult(name, url, False, f'status field not "ok": {data!r}', r.status_code, data)
        if data.get("service") != name:
            return CheckResult(
                name,
                url,
                False,
                f'service mismatch: expected {name!r}, got {data.get("service")!r}',
                r.status_code,
                data,
            )
        return CheckResult(name, url, True, "ok", r.status_code, data)
    except httpx.RequestError as e:
        return CheckResult(name, url, False, str(e), None, None)


def smoke_ask_gateway(client: httpx.Client, gateway_url: str) -> CheckResult:
    url = f"{gateway_url.rstrip('/')}/v1/ask"
    payload = {
        "mode": "strict",
        "note_text": "55-year-old with hypertension on lisinopril.",
        "question": "Summarize cardiovascular risk.",
    }
    try:
        r = client.post(url, json=payload, timeout=120.0)
        if r.status_code != 200:
            return CheckResult("gateway /v1/ask", url, False, f"HTTP {r.status_code}", r.status_code, r.text[:500])
        data = r.json()
        if data.get("status") != "ok":
            return CheckResult(
                "gateway /v1/ask",
                url,
                False,
                f'response status not ok: {data.get("status")!r}',
                r.status_code,
                data,
            )
        if not isinstance(data.get("answer"), str) or not data["answer"]:
            return CheckResult(
                "gateway /v1/ask",
                url,
                False,
                "missing or empty answer",
                r.status_code,
                data,
            )
        return CheckResult("gateway /v1/ask", url, True, "ok", r.status_code, None)
    except httpx.RequestError as e:
        return CheckResult("gateway /v1/ask", url, False, str(e), None, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check platform HTTP endpoints.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also POST /v1/ask on the gateway (slower; needs full stack healthy).",
    )
    args = parser.parse_args()

    bases: dict[str, str] = {}
    for name, (env_key, default) in DEFAULTS.items():
        bases[name] = _service_base(env_key, default)

    results: list[CheckResult] = []

    with httpx.Client() as client:
        for name, base in bases.items():
            results.append(check_health(client, name, base))

        if args.smoke:
            results.append(smoke_ask_gateway(client, bases["gateway-api"]))

    all_ok = True
    for res in results:
        status = "OK " if res.ok else "FAIL"
        extra = "" if res.ok else f" ({res.detail})"
        print(f"{status}  {res.name:22}  {res.url}{extra}")
        if not res.ok:
            all_ok = False
            if res.body is not None and isinstance(res.body, (dict, list)):
                print(f"       body: {json.dumps(res.body, indent=2)[:800]}")

    if not all_ok:
        print("\nFix: ensure `make up` (or docker compose) is running and services are healthy.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
