"""Phase 4 GPU governor (§11): FREE / BUSY / GAMING from NVML + Docker, published to Redis.

Run as its own container:  python -m app.governor
`Governor.step()` is a pure state machine (unit-tested); `main()` collects samples and publishes.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import config, state
from .settings import get_settings

log = logging.getLogger("gpu-governor")

STATES = ("FREE", "BUSY", "GAMING")
GIB = 2**30
OVERRIDE_KEY = "gpu_state_override"      # host/gpu-state.sh: manual FREE|BUSY|GAMING until cleared
HEARTBEAT_KEY = "gpu_governor:heartbeat"
TRANSITIONS_KEY = "gpu_governor:transitions"


@dataclass
class Sample:
    t: float
    util: int            # GPU util %
    enc: int             # NVENC util %
    total_gb: float
    used_gb: float
    ollama_gb: float     # VRAM held by Ollama models (the harness's own; counted as free)
    steam: bool          # gaming container running
    own_jobs: bool       # harness jobs hold GPU leases (sem:gpu): their load is not "someone else's"


class Governor:
    def __init__(self, cfg: dict[str, Any], current: str = "FREE"):
        self.cfg = cfg
        self.state = current if current in STATES else "FREE"
        self.gaming_seen: float | None = None
        self.util_high_since: float | None = None
        self.calm_since: float | None = None
        self.reason = ""

    def _c(self, key: str, default: float) -> float:
        return float(self.cfg.get(key, default))

    def step(self, s: Sample) -> str:
        util = 0 if s.own_jobs else s.util
        gaming = s.steam and (s.enc > 0 or util > self._c("gaming_util", 30))
        if gaming:
            self.gaming_seen = s.t
        if self.state == "GAMING" and self.gaming_seen is None:
            self.gaming_seen = s.t  # started in GAMING (restart): hold for the exit window
        if gaming or (self.state == "GAMING" and s.t - self.gaming_seen < self._c("gaming_exit_s", 300)):
            if gaming:
                self.reason = f"{self.cfg.get('gaming_container', 'steam-headless')} running, nvenc={s.enc}% util={util}%"
            self.state, self.util_high_since, self.calm_since = "GAMING", None, None
            return self.state

        free_gb = s.total_gb - s.used_gb + s.ollama_gb
        busy_util = self._c("busy_util", 70)
        if util > busy_util:
            self.util_high_since = s.t if self.util_high_since is None else self.util_high_since
        else:
            self.util_high_since = None
        low_vram = free_gb < self._c("busy_free_vram_gb", 8)
        hot = self.util_high_since is not None and s.t - self.util_high_since >= self._c("busy_enter_s", 30)
        if low_vram or hot:
            self.reason = f"free VRAM {free_gb:.1f} GB" if low_vram else f"util {util}% for {self._c('busy_enter_s', 30):.0f}s"
            self.state, self.calm_since = "BUSY", None
        elif self.state == "BUSY":
            if util <= busy_util:
                self.calm_since = s.t if self.calm_since is None else self.calm_since
                if s.t - self.calm_since >= self._c("busy_exit_s", 120):
                    self.state, self.reason, self.calm_since = "FREE", "below BUSY thresholds", None
            else:
                self.calm_since = None
        else:
            if self.state != "FREE":
                self.reason = "gaming ended"
            self.state = "FREE"
        return self.state


# ---------------------------------------------------------------- collectors
class Nvml:
    def __init__(self) -> None:
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def read(self) -> tuple[int, int, float, float]:
        nv, h = self.nv, self.h
        util = nv.nvmlDeviceGetUtilizationRates(h).gpu
        enc = nv.nvmlDeviceGetEncoderUtilization(h)[0]
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        return int(util), int(enc), mem.total / GIB, mem.used / GIB


async def steam_running(http: httpx.AsyncClient, name: str) -> bool:
    resp = await http.get(f"{get_settings().docker_host}/containers/json",
                          params={"filters": json.dumps({"name": [f"^/{name}$"]})})
    resp.raise_for_status()
    return any(c.get("State") == "running" for c in resp.json())


async def ollama_models(http: httpx.AsyncClient) -> list[dict[str, Any]]:
    resp = await http.get(f"{get_settings().ollama_url}/api/ps")
    resp.raise_for_status()
    return resp.json().get("models") or []


async def unload_gpu_models(http: httpx.AsyncClient, models: list[dict[str, Any]]) -> list[str]:
    """Free VRAM for the game. CPU-only models (local_cpu, size_vram 0) stay loaded."""
    done = []
    for m in models:
        if int(m.get("size_vram") or 0) > 0:
            name = m.get("name") or m.get("model")
            await http.post(f"{get_settings().ollama_url}/api/generate", json={"model": name, "keep_alive": 0})
            done.append(name)
    return done


async def own_jobs(r, lease_s: int = 900) -> bool:
    return await r.zcount(state.GpuSemaphore.KEY, time.time() - lease_s, "+inf") > 0


async def publish(r, new: str, *, reason: str, override: bool, sample: dict[str, Any] | None, ttl_s: int) -> bool:
    prev = await r.get("gpu_state")
    changed = prev != new
    if changed:
        await r.set("gpu_state", new)
        await r.publish("gpu_state_changed", new)
        await r.lpush(TRANSITIONS_KEY, json.dumps({"from": prev, "to": new, "at": time.time(), "reason": reason,
                                                   "override": override}))
        await r.ltrim(TRANSITIONS_KEY, 0, 199)
        log.info("gpu_state %s -> %s (%s)", prev, new, reason)
    await r.set(HEARTBEAT_KEY, json.dumps({"at": time.time(), "state": new, "override": override,
                                           "sample": sample}), ex=ttl_s)
    return changed


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    r = state.get_redis()
    gov = Governor((config.policy().get("gpu_governor") or {}), await r.get("gpu_state") or "FREE")
    nvml: Nvml | None = None
    async with httpx.AsyncClient(timeout=3, trust_env=False) as http:
        while True:
            cfg = config.policy().get("gpu_governor") or {}
            gov.cfg = cfg
            poll = float(cfg.get("poll_s", 2))
            ttl = int(cfg.get("heartbeat_ttl_s", 30))
            try:
                if nvml is None:
                    nvml = Nvml()
                util, enc, total, used = nvml.read()
                models = await ollama_models(http)
                sample = Sample(t=time.monotonic(), util=util, enc=enc, total_gb=total, used_gb=used,
                                ollama_gb=sum(int(m.get("size_vram") or 0) for m in models) / GIB,
                                steam=await steam_running(http, cfg.get("gaming_container", "steam-headless")),
                                own_jobs=await own_jobs(r))
                auto = gov.step(sample)
                override = await r.get(OVERRIDE_KEY)
                new = override if override in STATES else auto
                await publish(r, new, reason=f"override ({auto} measured)" if override in STATES else gov.reason,
                              override=override in STATES,
                              sample={**sample.__dict__, "auto": auto}, ttl_s=ttl)
                if new == "GAMING" and (gone := await unload_gpu_models(http, models)):
                    log.info("unloaded from VRAM: %s", ", ".join(gone))
            except Exception as e:  # keep the last state; the missing/erroring heartbeat raises an alert
                log.warning("governor sample failed: %s: %s", type(e).__name__, e)
                if nvml is not None and type(e).__module__.startswith("pynvml"):
                    nvml = None
                await r.set(HEARTBEAT_KEY, json.dumps({"at": time.time(), "error": f"{type(e).__name__}: {e}"[:300]}),
                            ex=ttl)
            await asyncio.sleep(poll)


if __name__ == "__main__":
    asyncio.run(main())
