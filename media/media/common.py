"""Shared plumbing: env, policy, Postgres rows, and the api's Redis protocols (run events, gpu_state, kill switch,
cancel, GPU semaphore, spend). Keep these in step with api/app/state.py."""
import asyncio
import functools
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import redis.asyncio as aioredis
import yaml
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get("DATABASE_URL", "")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
POLICY_PATH = os.environ.get("POLICY_PATH", "/config/policy.yaml")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "")
FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/files"))

_policy: tuple[float, dict[str, Any]] | None = None
_redis: aioredis.Redis | None = None


def policy() -> dict[str, Any]:
    """policy.yaml, re-read when it changes (the api hot-reloads it too)."""
    global _policy
    try:
        mtime = os.path.getmtime(POLICY_PATH)
    except OSError:
        return _policy[1] if _policy else {}
    if not _policy or _policy[0] != mtime:
        with open(POLICY_PATH, encoding="utf-8") as f:
            _policy = (mtime, yaml.safe_load(f) or {})
    return _policy[1]


def cfg() -> dict[str, Any]:
    return policy().get("media") or {}


def sub(name: str) -> dict[str, Any]:
    return cfg().get(name) or {}


def file_path(sha: str) -> Path:
    return FILES_ROOT / sha[:2] / sha


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def set_redis(r: aioredis.Redis) -> None:  # tests
    global _redis
    _redis = r


# ---------------------------------------------------------------- attachments rows
async def _conn() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)


async def get_att(att_id: str) -> dict[str, Any] | None:
    async with await _conn() as c:
        cur = await c.execute("SELECT * FROM attachments WHERE id = %s", (att_id,))
        return await cur.fetchone()


async def find_cached(sha: str, exclude_id: str) -> dict[str, Any] | None:
    """A finished extraction of the same bytes that ran at full quality (no fallback path)."""
    async with await _conn() as c:
        cur = await c.execute(
            """SELECT * FROM attachments WHERE sha256 = %s AND id <> %s AND status = 'done'
                 AND COALESCE((meta->>'degraded')::boolean, false) = false
               ORDER BY updated_at DESC LIMIT 1""", (sha, exclude_id))
        return await cur.fetchone()


async def update_att(att_id: str, **fields: Any) -> None:
    if "meta" in fields:
        fields["meta"] = json.dumps(fields["meta"], ensure_ascii=False, default=str)
    sets = ", ".join(f"{k} = %({k})s" + ("::jsonb" if k == "meta" else "") for k in fields)
    async with await _conn() as c:
        await c.execute(f"UPDATE attachments SET {sets}, updated_at = now() WHERE id = %(id)s", {**fields, "id": att_id})


# ---------------------------------------------------------------- Redis protocols (api/app/state.py)
async def emit(run_id: str, type_: str, data: Any = None) -> None:
    r, key = get_redis(), f"run:{run_id}"
    await r.xadd(key, {"type": type_, "data": json.dumps(data, ensure_ascii=False, default=str)}, maxlen=20000)
    await r.expire(key, 3600)


async def set_run_meta(run_id: str, **fields: Any) -> None:
    key = f"runmeta:{run_id}"
    await get_redis().hset(key, mapping={k: json.dumps(v, default=str) for k, v in fields.items()})
    await get_redis().expire(key, 7 * 86400)


async def is_cancelled(run_id: str) -> bool:
    r = get_redis()
    if await r.exists(f"cancel:{run_id}"):
        return True
    return json.loads(await r.hget(f"runmeta:{run_id}", "status") or "null") == "cancelled"


async def gpu_state() -> str:
    v = await get_redis().get("gpu_state")
    return v if v in ("FREE", "BUSY", "GAMING") else "FREE"


async def killswitch() -> str:
    v = await get_redis().get("killswitch")
    return v if v in ("off", "cloud", "all") else "off"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


async def budget_left(provider: str) -> bool:
    limit = (policy().get("budgets_usd_daily") or {}).get(provider)
    if limit is None:
        return False
    return float(await get_redis().get(f"spend:{_today()}:{provider}") or 0) < float(limit)


async def add_spend(provider: str, usd: float) -> None:
    if usd > 0:
        key = f"spend:{_today()}:{provider}"
        await get_redis().incrbyfloat(key, usd)
        await get_redis().expire(key, 7 * 86400)


class GpuSemaphore:
    """Same zset protocol as the api: at most `gpu_concurrency` GPU users across all workers."""
    KEY = "sem:gpu"

    def __init__(self, lease_s: int = 900, wait_s: float = 300):
        self.limit = int(policy().get("gpu_concurrency", 2))
        self.lease_s, self.wait_s, self.token = lease_s, wait_s, uuid.uuid4().hex

    async def __aenter__(self) -> "GpuSemaphore":
        r, deadline = get_redis(), time.monotonic() + self.wait_s
        while True:
            now = time.time()
            await r.zremrangebyscore(self.KEY, "-inf", now - self.lease_s)
            await r.zadd(self.KEY, {self.token: now})
            rank = await r.zrank(self.KEY, self.token)
            if rank is not None and rank < self.limit:
                return self
            await r.zrem(self.KEY, self.token)
            if time.monotonic() > deadline:
                raise TimeoutError("GPU semaphore wait timed out")
            await asyncio.sleep(0.5)

    async def __aexit__(self, *exc: Any) -> None:
        await get_redis().zrem(self.KEY, self.token)


# ---------------------------------------------------------------- text
@functools.cache
def _opencc(conf: str):
    import opencc
    return opencc.OpenCC(conf)


def to_trad(text: str, conf: str = "s2twp") -> str:
    """Simplified → Taiwan Traditional (the owner reads zh-TW; Whisper and Qwen drift to Simplified)."""
    return _opencc(conf).convert(text) if text else text


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"
