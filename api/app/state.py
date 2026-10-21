"""Redis-backed shared state: kill switch, gpu_state, dedupe, budget, run events, cancel, GPU semaphore."""
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

from .models import Budget, GpuState, KillSwitch
from .settings import get_settings

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


def set_redis(r: aioredis.Redis) -> None:  # tests
    global _redis
    _redis = r


# ---------------------------------------------------------------- kill switch / GPU
async def get_killswitch(r: aioredis.Redis) -> KillSwitch:
    v = await r.get("killswitch")
    return v if v in ("off", "cloud", "all") else "off"  # type: ignore[return-value]


async def set_killswitch(r: aioredis.Redis, state: KillSwitch) -> None:
    await r.set("killswitch", state)


async def get_gpu_state(r: aioredis.Redis) -> GpuState:
    v = await r.get("gpu_state")
    return v if v in ("FREE", "BUSY", "GAMING") else "FREE"  # type: ignore[return-value]


# ---------------------------------------------------------------- dedupe
async def dedupe(r: aioredis.Redis, key: str, ttl_s: int = 86400) -> bool:
    """True if first time seen."""
    return bool(await r.set(f"dedupe:{key}", "1", nx=True, ex=ttl_s))


# ---------------------------------------------------------------- privacy stickiness
async def is_thread_private(r: aioredis.Redis, thread_id: str) -> bool:
    return bool(await r.exists(f"thread_private:{thread_id}"))


async def mark_thread_private(r: aioredis.Redis, thread_id: str) -> None:
    await r.set(f"thread_private:{thread_id}", "1")


# ---------------------------------------------------------------- budget
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


async def get_budget(r: aioredis.Redis, policy: dict[str, Any]) -> Budget:
    limits = {k: float(v) for k, v in (policy.get("budgets_usd_daily") or {}).items()}
    spend: dict[str, float] = {}
    for provider in limits:
        v = await r.get(f"spend:{_today()}:{provider}")
        spend[provider] = float(v) if v else 0.0
    return Budget(spend_usd=spend, limits_usd=limits)


async def add_spend(r: aioredis.Redis, provider: str, usd: float) -> None:
    key = f"spend:{_today()}:{provider}"
    await r.incrbyfloat(key, usd)
    await r.expire(key, 7 * 86400)


# ---------------------------------------------------------------- run events (Redis Streams: replayable, no subscribe race)
EVENT_TTL_S = 3600


async def emit(r: aioredis.Redis, run_id: str, type_: str, data: Any = None) -> None:
    key = f"run:{run_id}"
    await r.xadd(key, {"type": type_, "data": json.dumps(data, ensure_ascii=False, default=str)}, maxlen=20000)
    await r.expire(key, EVENT_TTL_S)


TERMINAL = {"done", "error", "cancelled"}


async def read_events(r: aioredis.Redis, run_id: str, last_id: str = "0", block_ms: int = 15000) -> AsyncIterator[tuple[str, str, Any]]:
    """Yield (id, type, data) until a terminal event; yields ('', 'ping', None) on idle."""
    key = f"run:{run_id}"
    while True:
        resp = await r.xread({key: last_id}, block=block_ms, count=200)
        if not resp:
            yield "", "ping", None
            continue
        for _, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                type_ = fields.get("type", "")
                data = json.loads(fields.get("data") or "null")
                yield entry_id, type_, data
                if type_ in TERMINAL:
                    return


# ---------------------------------------------------------------- run metadata / cancel
async def set_run_meta(r: aioredis.Redis, run_id: str, **fields: Any) -> None:
    key = f"runmeta:{run_id}"
    await r.hset(key, mapping={k: json.dumps(v, default=str) for k, v in fields.items()})
    await r.expire(key, 7 * 86400)


async def get_run_meta(r: aioredis.Redis, run_id: str) -> dict[str, Any] | None:
    raw = await r.hgetall(f"runmeta:{run_id}")
    return {k: json.loads(v) for k, v in raw.items()} if raw else None


async def request_cancel(r: aioredis.Redis, run_id: str) -> None:
    await r.set(f"cancel:{run_id}", "1", ex=86400)


async def is_cancelled(r: aioredis.Redis, run_id: str) -> bool:
    return bool(await r.exists(f"cancel:{run_id}"))


# ---------------------------------------------------------------- GPU semaphore (crash-safe: entries expire)
class GpuSemaphore:
    KEY = "sem:gpu"

    def __init__(self, r: aioredis.Redis, limit: int, lease_s: int = 900, wait_s: float = 120):
        self.r, self.limit, self.lease_s, self.wait_s = r, limit, lease_s, wait_s
        self.token = uuid.uuid4().hex

    async def __aenter__(self) -> "GpuSemaphore":
        deadline = time.monotonic() + self.wait_s
        while True:
            now = time.time()
            await self.r.zremrangebyscore(self.KEY, "-inf", now - self.lease_s)
            await self.r.zadd(self.KEY, {self.token: now})
            rank = await self.r.zrank(self.KEY, self.token)
            if rank is not None and rank < self.limit:
                return self
            await self.r.zrem(self.KEY, self.token)
            if time.monotonic() > deadline:
                raise TimeoutError("GPU semaphore wait timed out")
            await asyncio.sleep(0.5)

    async def __aexit__(self, *exc: Any) -> None:
        await self.r.zrem(self.KEY, self.token)
