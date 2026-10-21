import logging
from contextlib import asynccontextmanager
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from . import db, state
from .routes import internal, line, web
from .settings import get_settings
from .telemetry import setup_tracing

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing("agent-api")
    pool = await db.open_db()
    saver = AsyncPostgresSaver(conn=pool)
    await saver.setup()
    app.state.checkpointer = saver
    arq = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    app.state.arq = arq

    async def enqueue(fn: str, payload: dict[str, Any], job_id: str, queue: str) -> None:
        await arq.enqueue_job(fn, payload, _job_id=job_id, _queue_name=queue)

    app.state.enqueue = enqueue
    yield
    await arq.aclose()
    await db.close_db()


app = FastAPI(title="agent-harness", lifespan=lifespan)
app.include_router(web.router)
app.include_router(line.router)
app.include_router(internal.router)


@app.get("/healthz")
async def healthz():
    await state.get_redis().ping()
    async with db.get_pool().connection() as conn:
        await conn.execute("SELECT 1")
    return {"ok": True}
