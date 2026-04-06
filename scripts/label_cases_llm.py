#!/usr/bin/env python3
"""Label synthetic cases by calling llm-service ``POST /v1/generate`` (temperature 0).

Reads ``data/processed/synthetic_cases.jsonl`` and writes ``data/processed/labeled_cases.jsonl``.

Requires ``LLM_BASE_URL`` (e.g. ``http://localhost:8060``) or pass ``--base-url``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

TRACE_ID_HEADER = "X-Trace-Id"

EXPECTED_SCHEMA_JSON = """{
  "label": "low" | "medium" | "high",
  "score": <float, 0.0 through 1.0 inclusive>,
  "red_flags": [<string>, ...],
  "reason": "<short string, clinical-style justification>"
}"""


class LLMJudgeResult(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    red_flags: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def _label_ok(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s not in ("low", "medium", "high"):
            raise ValueError("label must be low, medium, or high")
        return s

    @field_validator("red_flags", mode="before")
    @classmethod
    def _coerce_flags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("red_flags must be a list")
        return [str(x).strip() for x in v if str(x).strip()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip line {line_no}: {e}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                print(f"skip line {line_no}: not a JSON object", file=sys.stderr)
    return rows


def _compact_case_for_prompt(case: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "note_text",
        "structured_features",
        "entities",
        "generation_type",
        "source_doc_id",
    )
    return {k: case[k] for k in keys if k in case}


def _build_prompt(case: dict[str, Any]) -> str:
    payload = _compact_case_for_prompt(case)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""You are a clinical risk reviewer judging **synthetic training cases** (not real patients).

Task: Assign an overall acuity / concern level for triage-style training.

Respond with **only** one JSON object (no markdown fences, no prose before or after) that matches this schema exactly:
{EXPECTED_SCHEMA_JSON}

Guidance:
- "low": stable, routine, or minimal concern for acute harm.
- "medium": needs timely evaluation or has meaningful comorbidity / symptom burden.
- "high": urgent red-flag patterns, severe symptoms, or dangerous combinations (even if synthetic).
- "score": 0.0–1.0 aligned with label (low ~0.0–0.35, medium ~0.35–0.7, high ~0.7–1.0).
- "red_flags": short strings (empty list allowed if none).
- "reason": one concise sentence.

CASE (JSON):
{body}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("empty model text")
    fence = re.search(
        r"```(?:json)?\s*([\s\S]*?)```",
        t,
        re.IGNORECASE,
    )
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model text")
    raw = t[start : end + 1]
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


def _call_generate(
    client: httpx.Client,
    base_url: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    trace_id: str,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/generate"
    payload = {
        "trace_id": trace_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = client.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            TRACE_ID_HEADER: trace_id,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    if data.get("status", "ok") != "ok":
        raise RuntimeError(f"LLM status not ok: {data!r}")
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError("LLM response missing text")
    return text


def _label_once(
    client: httpx.Client,
    base_url: str,
    case: dict[str, Any],
    *,
    max_tokens: int,
    temperature: float,
) -> LLMJudgeResult:
    prompt = _build_prompt(case)
    trace_id = str(uuid.uuid4())
    raw = _call_generate(
        client,
        base_url,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        trace_id=trace_id,
    )
    obj = _extract_json_object(raw)
    return LLMJudgeResult.model_validate(obj)


def _label_with_retries(
    client: httpx.Client,
    base_url: str,
    case: dict[str, Any],
    *,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep_sec: float,
) -> LLMJudgeResult:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _label_once(
                client,
                base_url,
                case,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(retry_sleep_sec * attempt)
    assert last_err is not None
    raise last_err


def _merge_labeled(case: dict[str, Any], judge: LLMJudgeResult) -> dict[str, Any]:
    out = dict(case)
    out["label"] = judge.label
    out["score"] = judge.score
    out["red_flags"] = judge.red_flags
    out["reason"] = judge.reason
    out["label_source"] = "llm"
    return out


def _process_batch(
    client: httpx.Client,
    base_url: str,
    indexed_cases: list[tuple[int, dict[str, Any]]],
    *,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep_sec: float,
    workers: int,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    errors: dict[int, BaseException] = {}

    def _work(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        idx, case = item
        judge = _label_with_retries(
            client,
            base_url,
            case,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_sleep_sec=retry_sleep_sec,
        )
        return idx, _merge_labeled(case, judge)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_work, it): it[0] for it in indexed_cases}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                i, labeled = fut.result()
                results[i] = labeled
            except BaseException as exc:
                errors[idx] = exc

    if errors:
        first = min(errors)
        raise RuntimeError(
            f"labeling failed for index {first} (and {len(errors) - 1} other(s)): {errors[first]}"
        ) from errors[first]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/synthetic_cases.jsonl"),
        help="Input JSONL (default: data/processed/synthetic_cases.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/labeled_cases.jsonl"),
        help="Output JSONL (default: data/processed/labeled_cases.jsonl)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("LLM_BASE_URL", "http://localhost:8060"),
        help="llm-service base URL (default: env LLM_BASE_URL or http://localhost:8060)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Cases per batch for progress and parallel chunking (default: 16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel HTTP workers per batch (default: 2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="LLM max_new_tokens (capped by service; default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per case on invalid JSON or HTTP errors (default: 4)",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=0.75,
        help="Base sleep seconds between retries (linear backoff; default: 0.75)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP client timeout seconds (default: 300)",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output
    base_url = args.base_url.strip().rstrip("/")

    if not base_url.startswith("http"):
        raise SystemExit("--base-url must start with http(s)://")

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    cases = _read_jsonl(in_path)
    if not cases:
        raise SystemExit(f"No cases in {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cases)
    batch_size = max(1, int(args.batch_size))
    workers = max(1, int(args.workers))

    print(
        f"Labeling {n} cases via {base_url}/v1/generate "
        f"(batch_size={batch_size}, workers={workers}, temperature={args.temperature})",
        flush=True,
    )

    all_labeled: dict[int, dict[str, Any]] = {}

    with httpx.Client(timeout=args.timeout) as client:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = [(i, cases[i]) for i in range(start, end)]
            t0 = time.perf_counter()
            batch_results = _process_batch(
                client,
                base_url,
                chunk,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retries=args.max_retries,
                retry_sleep_sec=args.retry_sleep,
                workers=workers,
            )
            for i in range(start, end):
                all_labeled[i] = batch_results[i]
            dt = time.perf_counter() - t0
            print(
                f"progress: labeled {end}/{n} cases "
                f"(batch {start + 1}-{end} in {dt:.1f}s)",
                flush=True,
            )

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for i in range(n):
            out.write(json.dumps(all_labeled[i], ensure_ascii=False) + "\n")

    print(f"Wrote {n} labeled rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
