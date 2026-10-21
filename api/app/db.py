"""Postgres: connection pool, schema, decision/usage/audit/alert logging."""
import json
import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .settings import get_settings

log = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    job_id          UUID NOT NULL,
    run_id          TEXT,
    thread_id       TEXT NOT NULL,
    channel         TEXT NOT NULL,
    mode            TEXT NOT NULL,
    redacted_text   TEXT,
    inputs          JSONB NOT NULL,      -- job meta, gate0, gpu_state, budget, killswitch
    router          JSONB NOT NULL,      -- raw router output (Jev / fallback)
    policy_version  TEXT,
    decision        JSONB NOT NULL,
    outcome         JSONB,               -- verify result, escalations, user correction, status
    latency_ms      INT,
    cost_usd        NUMERIC(12, 6) DEFAULT 0
);
CREATE INDEX IF NOT EXISTS decisions_created_at ON decisions (created_at);
CREATE INDEX IF NOT EXISTS decisions_run_id ON decisions (run_id);

CREATE TABLE IF NOT EXISTS usage (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id          TEXT,
    provider        TEXT NOT NULL,
    model_alias     TEXT,
    input_tokens    INT DEFAULT 0,
    output_tokens   INT DEFAULT 0,
    cost_usd        NUMERIC(12, 6) DEFAULT 0,
    credits         NUMERIC(12, 3) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id          TEXT,
    kind            TEXT NOT NULL,      -- tool_call | decision | killswitch | command
    name            TEXT,
    detail          JSONB
);

CREATE TABLE IF NOT EXISTS thread_meta (
    thread_id       TEXT PRIMARY KEY,
    hidden          BOOLEAN NOT NULL DEFAULT false,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    key             TEXT NOT NULL,
    message         TEXT NOT NULL,
    delivered       BOOLEAN DEFAULT false
);
"""


def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
    return _pool


async def open_db() -> AsyncConnectionPool:
    pool = get_pool()
    await pool.open(wait=True, timeout=60)
    async with pool.connection() as conn:
        await conn.execute(SCHEMA, prepare=False)
    return pool


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


async def log_decision(**f: Any) -> None:
    try:
        async with get_pool().connection() as conn:
            await conn.execute(
                """INSERT INTO decisions (job_id, run_id, thread_id, channel, mode, redacted_text, inputs, router,
                                          policy_version, decision, latency_ms)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (f["job_id"], f["run_id"], f["thread_id"], f["channel"], f["mode"], f["redacted_text"],
                 _j(f["inputs"]), _j(f["router"]), f["policy_version"], _j(f["decision"]), f["latency_ms"]),
            )
    except Exception:
        log.exception("failed to log decision")


async def update_outcome(run_id: str, outcome: dict[str, Any], cost_usd: float = 0.0) -> None:
    try:
        async with get_pool().connection() as conn:
            await conn.execute(
                "UPDATE decisions SET outcome = %s, cost_usd = %s WHERE run_id = %s",
                (_j(outcome), cost_usd, run_id),
            )
    except Exception:
        log.exception("failed to update outcome")


async def log_usage(run_id: str, provider: str, alias: str, inp: int, out: int, cost: float) -> None:
    try:
        async with get_pool().connection() as conn:
            await conn.execute(
                "INSERT INTO usage (run_id, provider, model_alias, input_tokens, output_tokens, cost_usd) VALUES (%s,%s,%s,%s,%s,%s)",
                (run_id, provider, alias, inp, out, cost),
            )
    except Exception:
        log.exception("failed to log usage")


async def audit(kind: str, name: str | None, detail: dict[str, Any], run_id: str | None = None) -> None:
    try:
        async with get_pool().connection() as conn:
            await conn.execute(
                "INSERT INTO audit_log (run_id, kind, name, detail) VALUES (%s,%s,%s,%s)",
                (run_id, kind, name, _j(detail)),
            )
    except Exception:
        log.exception("failed to write audit log")


async def log_alert(key: str, message: str, delivered: bool) -> None:
    try:
        async with get_pool().connection() as conn:
            await conn.execute(
                "INSERT INTO alerts (key, message, delivered) VALUES (%s,%s,%s)", (key, message, delivered)
            )
    except Exception:
        log.exception("failed to log alert")


async def list_threads(prefixes: tuple[str, ...], limit: int = 100) -> list[dict[str, Any]]:
    """Threads seen in the decision log (newest activity first), minus ones hidden since their last message."""
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            """SELECT d.thread_id, max(d.created_at) AS updated_at,
                      (array_agg(d.redacted_text ORDER BY d.id)
                       FILTER (WHERE btrim(coalesce(d.redacted_text, '')) <> ''))[1] AS first_text
               FROM decisions d
               LEFT JOIN thread_meta m ON m.thread_id = d.thread_id
               WHERE d.thread_id LIKE ANY(%s)
               GROUP BY d.thread_id
               -- hidden only until the thread has newer activity (from any channel)
               HAVING NOT (bool_or(coalesce(m.hidden, false)) AND max(m.updated_at) >= max(d.created_at))
               ORDER BY updated_at DESC
               LIMIT %s""",
            ([f"{p}%" for p in prefixes], limit),
        )
        return await cur.fetchall()


async def set_thread_hidden(thread_id: str, hidden: bool) -> None:
    async with get_pool().connection() as conn:
        await conn.execute(
            """INSERT INTO thread_meta (thread_id, hidden) VALUES (%s, %s)
               ON CONFLICT (thread_id) DO UPDATE SET hidden = EXCLUDED.hidden, updated_at = now()""",
            (thread_id, hidden),
        )
