"""Async persistent trace storage for orchestrator /v1/ask requests."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import asyncpg

_pool: asyncpg.Pool | None = None
_init_lock = asyncio.Lock()


def _trace_db_enabled() -> bool:
    raw = os.getenv("ORCHESTRATOR_TRACE_DB_ENABLED", "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _postgres_dsn() -> str:
    dsn = os.getenv("ORCHESTRATOR_TRACE_DB_DSN", "").strip()
    if dsn:
        return dsn
    user = os.getenv("POSTGRES_USER", "clinical_ai")
    pwd = os.getenv("POSTGRES_PASSWORD", "clinical_ai")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db = os.getenv("POSTGRES_DB", "clinical_ai")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


CREATE_TRACE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS orchestrator_request_traces (
    trace_id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb
);
"""


UPSERT_TRACE_SQL = """
INSERT INTO orchestrator_request_traces (
    trace_id, query, mode, answer, sources, warnings
) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
ON CONFLICT (trace_id) DO UPDATE SET
    timestamp = NOW(),
    query = EXCLUDED.query,
    mode = EXCLUDED.mode,
    answer = EXCLUDED.answer,
    sources = EXCLUDED.sources,
    warnings = EXCLUDED.warnings;
"""


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    async with _init_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=_postgres_dsn(),
                min_size=1,
                max_size=5,
                command_timeout=5.0,
            )
            async with _pool.acquire() as conn:
                await conn.execute(CREATE_TRACE_TABLE_SQL)
    return _pool


async def save_request_trace(
    *,
    trace_id: str,
    query: str,
    mode: str,
    answer: str,
    sources: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[bool, str | None]:
    """
    Persist one request trace.

    Returns ``(ok, error_message)`` and never raises.
    """
    if not _trace_db_enabled():
        return False, "trace_db_disabled"
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                UPSERT_TRACE_SQL,
                trace_id,
                query,
                mode,
                answer,
                json.dumps(sources, separators=(",", ":")),
                json.dumps(warnings, separators=(",", ":")),
            )
        return True, None
    except Exception as exc:  # pragma: no cover - best effort telemetry path
        return False, str(exc)
