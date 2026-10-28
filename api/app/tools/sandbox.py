"""Client for sandbox-manager (§9a). The coding graph only talks to sandboxes through this."""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import yaml

from ..settings import get_settings


class SandboxError(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    pass


def coding_config() -> dict[str, Any]:
    with open(os.environ.get("CODING_CONFIG_PATH", "/config/coding.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SandboxClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 120):
        self.base_url = base_url or os.environ.get("SANDBOX_MANAGER_URL", "http://sandbox-manager:8200")
        self.token = token if token is not None else get_settings().harness_token
        self.timeout = timeout

    async def _call(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, trust_env=False,
                                     headers={"Authorization": f"Bearer {self.token}"}) as c:
            r = await c.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:300]
            raise SandboxError(f"sandbox-manager {method} {path}: {r.status_code} {detail}")
        return r.json()

    async def create(self, task_id: str, repo: str) -> dict[str, Any]:
        return await self._call("POST", "/tasks", json={"task_id": task_id, "repo": repo})

    async def start(self, task_id: str, kind: str, prompt: str | None = None) -> str:
        return (await self._call("POST", f"/tasks/{task_id}/jobs", json={"kind": kind, "prompt": prompt}))["job_id"]

    async def status(self, task_id: str, job_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/tasks/{task_id}/jobs/{job_id}")

    async def kill(self, task_id: str, job_id: str) -> None:
        await self._call("DELETE", f"/tasks/{task_id}/jobs/{job_id}")

    async def diff(self, task_id: str, max_chars: int = 60_000) -> dict[str, Any]:
        return await self._call("GET", f"/tasks/{task_id}/diff", params={"max_chars": max_chars})

    async def apply(self, task_id: str, branch: str, message: str) -> dict[str, Any]:
        return await self._call("POST", f"/tasks/{task_id}/apply", json={"branch": branch, "message": message})

    async def delete(self, task_id: str) -> None:
        await self._call("DELETE", f"/tasks/{task_id}")

    async def run_job(self, task_id: str, kind: str, prompt: str | None = None, *, poll_s: float = 3,
                      timeout_s: float = 1200, cancelled: Callable[[], Awaitable[bool]] | None = None
                      ) -> dict[str, Any]:
        """Start a job and wait for it; the container is killed on cancel, timeout or task abort."""
        job = await self.start(task_id, kind, prompt)
        deadline = time.monotonic() + timeout_s
        done = False
        try:
            while True:
                res = await self.status(task_id, job)
                if res.get("status") == "exited":
                    done = True
                    return res
                if cancelled and await cancelled():
                    raise JobCancelled(f"{kind} cancelled")
                if time.monotonic() > deadline:
                    raise SandboxError(f"{kind} timed out after {int(timeout_s)} s")
                await asyncio.sleep(poll_s)
        finally:
            if not done:
                with contextlib.suppress(Exception):
                    await asyncio.shield(self.kill(task_id, job))
