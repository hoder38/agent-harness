"""Owner alerts → LINE push (deduplicated) + `alerts` table; health watcher cron (§19)."""
import json
import logging
import time
from typing import Any

import httpx

from .. import config, db, state
from ..channels import line
from ..settings import get_settings

log = logging.getLogger(__name__)


async def send_alert(key: str, message: str, *, dedupe_s: int | None = None) -> bool:
    """Send at most one alert per `key` per dedupe window. Returns True if sent."""
    r = state.get_redis()
    dedupe_s = dedupe_s or int(config.policy().get("alerts", {}).get("dedupe_s", 3600))
    if not await r.set(f"alert:{key}", "1", nx=True, ex=dedupe_s):
        return False
    s = get_settings()
    delivered = False
    if s.line_owner_user_id and s.line_channel_access_token:
        try:
            await line.push(s.line_owner_user_id, f"⚠️ {message}")
            delivered = True
        except Exception:
            log.exception("LINE alert push failed")
    log.warning("ALERT %s: %s", key, message)
    await db.log_alert(key, message, delivered)
    return True


async def check_containers() -> list[tuple[str, str]]:
    s = get_settings()
    filters = json.dumps({"label": [f"com.docker.compose.project={s.compose_project}"]})
    async with httpx.AsyncClient(base_url=s.docker_host, timeout=5, trust_env=False) as c:
        resp = await c.get("/containers/json", params={"all": "1", "filters": filters})
        resp.raise_for_status()
        containers = resp.json()
    issues = []
    for ct in containers:
        svc = ct.get("Labels", {}).get("com.docker.compose.service", ct.get("Names", ["?"])[0])
        if ct.get("Labels", {}).get("com.docker.compose.oneoff") == "True":
            continue
        st, status = ct.get("State"), ct.get("Status", "")
        if st != "running":
            issues.append((f"container:{svc}:down", f"{svc} is {st} ({status})"))
        elif "(unhealthy)" in status:
            issues.append((f"container:{svc}:unhealthy", f"{svc} is unhealthy"))
    return issues


async def check_queues(pol: dict[str, Any]) -> list[tuple[str, str]]:
    r = state.get_redis()
    max_age = float(pol.get("alerts", {}).get("queue_age_s", 300))
    issues = []
    now_ms = time.time() * 1000
    for q in {v["name"] for v in pol.get("queues", {}).values()}:
        oldest = await r.zrange(q, 0, 0, withscores=True)
        if oldest and now_ms - oldest[0][1] > max_age * 1000:
            age = int((now_ms - oldest[0][1]) / 1000)
            issues.append((f"queue:{q}:stale", f"queue {q} has a job waiting {age}s"))
    return issues


async def check_budget(pol: dict[str, Any]) -> list[tuple[str, str]]:
    r = state.get_redis()
    b = await state.get_budget(r, pol)
    issues = []
    frac = b.total_fraction()
    if frac >= 1.0:
        if await state.get_killswitch(r) == "off":
            await state.set_killswitch(r, "cloud")
            await db.audit("killswitch", "cloud", {"source": "budget", "fraction": frac})
        issues.append(("budget:100", f"daily spend at {frac:.0%} of budget → killswitch=cloud"))
    elif frac >= float(pol.get("alert_budget_fraction", 0.8)):
        issues.append(("budget:80", f"daily spend at {frac:.0%} of budget"))
    return issues


async def alert_watcher(ctx: dict[str, Any]) -> None:
    pol = config.policy()
    issues: list[tuple[str, str]] = []
    for check in (check_containers(), check_queues(pol), check_budget(pol)):
        try:
            issues += await check
        except Exception as e:
            issues.append((f"watcher:{type(e).__name__}", f"health check failed: {e}"))
    for key, msg in issues:
        await send_alert(key, msg)
