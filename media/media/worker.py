"""arq worker for queue `arq:media` (§5): extract attachments, then hand the run to the agent worker."""
import logging
from typing import Any

from arq import Retry, cron, func
from arq.connections import RedisSettings

from . import asr, common
from .extract import extract

_h = logging.StreamHandler()   # our own logger only: arq already configures its own
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger("media").addHandler(_h)
logging.getLogger("media").setLevel(logging.INFO)
log = logging.getLogger(__name__)

_queues = common.policy().get("queues") or {}
QUEUE = (_queues.get("media") or {}).get("name", "arq:media")


async def extract_attachment(ctx: dict[str, Any], att_id: str) -> dict[str, Any]:
    """Eager: starts right after the upload, so the text is often ready when the owner presses send."""
    att = await extract(att_id)
    return {"status": att["status"], "kind": att["kind"]}


async def media_run(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    run_id = payload["run_id"]
    job = payload["job"]
    atts = job.get("attachments") or []
    heavy_kinds = set(common.policy().get("heavy_media_kinds") or ["video", "large_doc"])
    if await common.killswitch() == "all" and any(a["kind"] in heavy_kinds for a in atts):
        await common.emit(run_id, "status", {"status": "paused", "reason": "killswitch=all"})
        raise Retry(defer=int(common.policy().get("defer_recheck_s", 300)))

    kinds = []
    for i, a in enumerate(atts):
        if await common.is_cancelled(run_id):
            return {"status": "cancelled"}

        async def progress(step: str, a=a) -> None:
            await common.emit(run_id, "status", {"status": "processing", "step": step, "attachment": a["id"]})

        await common.emit(run_id, "status", {"status": "processing", "step": f"extracting {i + 1}/{len(atts)}",
                                             "attachment": a["id"]})
        row = await extract(a["id"], progress)
        kinds.append(row["kind"])
        await common.emit(run_id, "attachment", {"id": row["id"], "kind": row["kind"], "status": row["status"],
                                                 "summary": row.get("summary"), "error": row.get("error")})
    if await common.is_cancelled(run_id):
        return {"status": "cancelled"}
    q = "background" if any(k in heavy_kinds for k in kinds) else "interactive"
    qname = (_queues.get(q) or {}).get("name", f"arq:{q}")
    await ctx["redis"].enqueue_job("submit_media_run", payload, _job_id=f"msub:{run_id}", _queue_name=qname)
    await common.emit(run_id, "status", {"status": "queued", "step": "attachments ready"})
    return {"status": "submitted", "kinds": kinds}


async def unload_idle(ctx: dict[str, Any]) -> None:
    """Free VRAM: whisper leaves the GPU when idle or when the GPU stops being FREE (§11)."""
    asr.unload_idle(await common.gpu_state())


class Settings:
    redis_settings = RedisSettings.from_dsn(common.REDIS_URL)
    queue_name = QUEUE
    functions = [func(extract_attachment, max_tries=2),
                 func(media_run, max_tries=50)]   # Retry while the kill switch pauses heavy media
    cron_jobs = [cron(unload_idle, second={0, 30}, run_at_startup=False)]
    max_jobs = int((common.cfg().get("max_jobs") or 2))
    job_timeout = int((_queues.get("media") or {}).get("timeout_s", 3600))
    max_tries = 3
    allow_abort_jobs = True
    health_check_interval = 30
