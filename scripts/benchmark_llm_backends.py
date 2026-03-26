#!/usr/bin/env python3
"""
Benchmark llm-service backends via /v1/generate.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PROMPTS = [
    "Summarize key clinical risks in 3 short bullet points.",
    "Given chest pain, diabetes, and hypertension, provide a concise risk-focused plan.",
    "List likely monitoring steps from provided evidence only, in plain sentences.",
]


@dataclass
class RequestResult:
    ok: bool
    latency_ms: float
    completion_tokens: int
    status_code: int | None
    error: str | None


def parse_backend(arg: str) -> tuple[str, str]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Expected --backend name=http://host:port")
    name, url = arg.split("=", 1)
    name = name.strip()
    url = url.strip().rstrip("/")
    if not name or not url.startswith("http"):
        raise argparse.ArgumentTypeError("Expected --backend name=http://host:port")
    return name, url


def load_prompts(prompts_file: str | None) -> list[str]:
    if not prompts_file:
        return list(DEFAULT_PROMPTS)
    lines = [
        ln.strip()
        for ln in Path(prompts_file).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        raise ValueError(f"No prompts found in {prompts_file}")
    return lines


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def run_one_request(
    client: httpx.Client,
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    url = f"{base_url}/v1/generate"
    payload = {
        "trace_id": str(uuid.uuid4()),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    try:
        resp = client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            return RequestResult(False, elapsed_ms, 0, resp.status_code, f"HTTP {resp.status_code}")
        data = resp.json()
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return RequestResult(True, elapsed_ms, completion_tokens, resp.status_code, None)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(False, elapsed_ms, 0, None, str(exc))


def benchmark_backend(
    name: str,
    base_url: str,
    prompts: list[str],
    warmup: int,
    runs: int,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_sec) as client:
        for _ in range(warmup):
            for prompt in prompts:
                _ = run_one_request(client, base_url, prompt, max_tokens, temperature)
        measured = [
            run_one_request(client, base_url, prompt, max_tokens, temperature)
            for _ in range(runs)
            for prompt in prompts
        ]

    ok = [r for r in measured if r.ok]
    latencies = [r.latency_ms for r in ok]
    total_tokens = sum(r.completion_tokens for r in ok)
    total_sec = sum(latencies) / 1000.0 if latencies else 0.0
    avg = statistics.mean(latencies) if latencies else 0.0
    p50 = percentile(latencies, 0.5)
    p95 = percentile(latencies, 0.95)
    tps = (total_tokens / total_sec) if total_sec > 0 else 0.0

    return {
        "backend": name,
        "base_url": base_url,
        "warmup_requests": warmup * len(prompts),
        "measured_requests": runs * len(prompts),
        "successful_requests": len(ok),
        "failed_requests": len(measured) - len(ok),
        "first_token_latency_ms": None,
        "first_token_latency_note": "not_available_non_streaming_endpoint",
        "total_latency_ms": sum(latencies),
        "average_latency_ms": avg,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "tokens_generated": total_tokens,
        "approx_tokens_per_sec": tps,
        "errors": [{"status_code": r.status_code, "error": r.error} for r in measured if not r.ok],
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nLLM Backend Benchmark Summary")
    print("-" * 72)
    for r in results:
        print(f"Backend: {r['backend']} ({r['base_url']})")
        print(
            f"  Requests: ok={r['successful_requests']}/{r['measured_requests']} "
            f"(warmup={r['warmup_requests']}, failed={r['failed_requests']})"
        )
        print(
            f"  Latency: avg={r['average_latency_ms']:.1f} ms, "
            f"p50={r['p50_latency_ms']:.1f} ms, p95={r['p95_latency_ms']:.1f} ms, "
            f"total={r['total_latency_ms']:.1f} ms"
        )
        print(
            f"  Tokens: generated={r['tokens_generated']}, "
            f"approx_tps={r['approx_tokens_per_sec']:.2f}"
        )
        print(
            f"  First token latency: {r['first_token_latency_ms']} "
            f"({r['first_token_latency_note']})"
        )
        if r["errors"]:
            e = r["errors"][0]
            print(f"  Errors: {len(r['errors'])} (first: status={e['status_code']} err={e['error']})")
        print("-" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark llm-service backends")
    parser.add_argument("--backend", action="append", required=True, type=parse_backend)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--output", default="llm_benchmark_results.json")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    results = [
        benchmark_backend(
            name=n,
            base_url=u,
            prompts=prompts,
            warmup=args.warmup,
            runs=args.runs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
        )
        for n, u in args.backend
    ]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "prompts_count": len(prompts),
            "warmup": args.warmup,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout_sec": args.timeout_sec,
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nSaved JSON results to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
