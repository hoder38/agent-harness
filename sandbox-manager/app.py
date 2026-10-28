"""sandbox-manager (§9a): the only service that can create containers (via socket-proxy-rw).

A task = a `git clone --local` of an allowlisted repo under SANDBOX_ROOT/<task>/repo. Jobs run in throwaway
containers that mount only that clone: `copilot` (Copilot CLI headless, token injected here only) and `test`
(per-repo test command, never sees the token). `apply` squashes the clone's changes into one commit and
fetches it into the original repo as a new local branch; nothing is ever pushed.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import docker
import yaml
from docker.errors import NotFound
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

log = logging.getLogger("sandbox-manager")
logging.basicConfig(level=logging.INFO)

ROOT = Path(os.environ.get("SANDBOX_ROOT", "/srv/agent/sandboxes"))
CONFIG_PATH = os.environ.get("CODING_CONFIG_PATH", "/config/coding.yaml")
TOKEN = os.environ.get("SANDBOX_MANAGER_TOKEN", "")
_TASK_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,63}$")
_BRANCH_RE = re.compile(r"^agent/[A-Za-z0-9._/-]{1,80}$")
LABEL = "agent-harness.sandbox"
OUT_MAX = 200_000


def cfg() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sb() -> dict[str, Any]:
    return cfg().get("sandbox") or {}


def repo_cfg(name: str) -> dict[str, Any]:
    repo = (cfg().get("repos") or {}).get(name)
    if not repo:
        raise HTTPException(404, f"unknown repo {name!r}")
    return repo


_client: docker.DockerClient | None = None


def dk() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.DockerClient(base_url=os.environ.get("DOCKER_HOST", "tcp://socket-proxy-rw:2375"))
    return _client


def auth(authorization: str = Header("")) -> None:
    if not TOKEN or authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")


def git(cwd: Path | str, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=120)
    if check and p.returncode:
        raise HTTPException(500, f"git {' '.join(args[:2])}: {(p.stderr or p.stdout)[-500:]}")
    return p.stdout


def src_git(src: Path | str, *args: str, check: bool = True) -> str:
    """Git on an original repo: only its .git is mounted, so address it as a bare git dir."""
    p = subprocess.run(["git", "--git-dir", f"{src}/.git", *args], capture_output=True, text=True, timeout=120)
    if check and p.returncode:
        raise HTTPException(500, f"git {' '.join(args[:2])}: {(p.stderr or p.stdout)[-500:]}")
    return p.stdout


def task_dir(task_id: str) -> Path:
    if not _TASK_RE.match(task_id):
        raise HTTPException(400, "bad task id")
    return ROOT / task_id


def load_meta(task_id: str) -> dict[str, Any]:
    f = task_dir(task_id) / "meta.json"
    if not f.exists():
        raise HTTPException(404, "no such task")
    return json.loads(f.read_text())


def save_meta(task_id: str, meta: dict[str, Any]) -> None:
    (task_dir(task_id) / "meta.json").write_text(json.dumps(meta))


app = FastAPI(title="sandbox-manager")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "root": str(ROOT), "docker": bool(dk().ping())}


@app.get("/repos", dependencies=[Depends(auth)])
def repos() -> dict[str, Any]:
    return {"repos": {k: {"description": v.get("description", ""), "aliases": v.get("aliases") or []}
                      for k, v in (cfg().get("repos") or {}).items()}}


class CreateBody(BaseModel):
    task_id: str
    repo: str


@app.post("/tasks", dependencies=[Depends(auth)])
def create_task(body: CreateBody) -> dict[str, Any]:
    d = task_dir(body.task_id)
    if (d / "meta.json").exists():
        return load_meta(body.task_id)
    repo = repo_cfg(body.repo)
    src = Path(repo["path"])
    if not (src / ".git").exists():
        raise HTTPException(500, f"{src} is not a git repo (is it mounted?)")
    (d / "home" / ".copilot").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    clone = d / "repo"
    p = subprocess.run(["git", "clone", "--local", "--no-hardlinks", "--quiet", str(src), str(clone)],
                       capture_output=True, text=True, timeout=300)
    if p.returncode:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(500, f"clone failed: {p.stderr[-500:]}")
    try:
        git(clone, "remote", "remove", "origin")
        # Trust only the workspace; nothing else from the host's Copilot config is visible.
        (d / "home" / ".copilot" / "config.json").write_text(json.dumps({"trustedFolders": ["/workspace"]}))
        meta = {"task_id": body.task_id, "repo": body.repo, "src": str(src),
                "base": git(clone, "rev-parse", "HEAD").strip(),
                "branch": src_git(src, "rev-parse", "--abbrev-ref", "HEAD").strip(), "created": time.time(), "jobs": 0}
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    save_meta(body.task_id, meta)
    return meta


class JobBody(BaseModel):
    kind: str                       # "copilot" | "test"
    prompt: str | None = None


def _mounts(d: Path, mounts: dict[str, str]) -> dict[str, dict[str, str]]:
    out = {}
    for rel, target in (mounts or {}).items():
        host = (d / "repo" / rel).resolve()
        if not str(host).startswith(str(d / "repo")):
            raise HTTPException(400, f"mount {rel!r} escapes the clone")
        out[str(host)] = {"bind": target, "mode": "rw"}
    return out


def _limits(s: dict[str, Any]) -> dict[str, Any]:
    return {"mem_limit": str(s.get("mem", "6g")), "nano_cpus": int(float(s.get("cpus", 4)) * 1e9),
            "pids_limit": int(s.get("pids", 1024)), "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"], "tmpfs": {"/tmp": "size=1g"}}


def copilot_cmd(prompt: str, s: dict[str, Any]) -> list[str]:
    cmd = ["copilot", "-p", prompt, "-s", "--output-format", "json", "--no-ask-user", "--no-auto-update",
           "--disable-builtin-mcps", "--secret-env-vars=COPILOT_GITHUB_TOKEN",
           "--max-ai-credits", str(int(s.get("max_ai_credits", 2000)))]
    if s.get("model"):
        cmd += ["--model", str(s["model"])]
    for t in s.get("allow_tools") or []:
        cmd += ["--allow-tool", str(t)]
    for t in s.get("deny_tools") or []:
        cmd += ["--deny-tool", str(t)]
    return cmd


@app.post("/tasks/{task_id}/jobs", dependencies=[Depends(auth)])
def start_job(task_id: str, body: JobBody) -> dict[str, Any]:
    meta = load_meta(task_id)
    d = task_dir(task_id)
    s = sb()
    running = dk().containers.list(filters={"label": [LABEL]})
    if len(running) >= int(s.get("max_running", 2)):
        raise HTTPException(429, "too many sandboxes running")
    meta["jobs"] += 1
    name = f"sbx-{task_id[:24]}-{body.kind}-{meta['jobs']}"
    labels = {LABEL: "1", f"{LABEL}.task": task_id, f"{LABEL}.kind": body.kind, f"{LABEL}.started": str(int(time.time()))}
    proxy = str(s.get("proxy", "http://egress-proxy:4750"))
    if body.kind == "copilot":
        if not body.prompt:
            raise HTTPException(400, "prompt required")
        token = os.environ.get("COPILOT_GITHUB_TOKEN", "")
        if not token:
            raise HTTPException(503, "COPILOT_GITHUB_TOKEN is not set in .env")
        kw = dict(image=s.get("image", "agent-harness-sandbox:latest"), command=copilot_cmd(body.prompt, s),
                  working_dir="/workspace", user="1000:1000", network=s.get("network", "agent-sandbox"),
                  environment={"COPILOT_GITHUB_TOKEN": token, "HOME": "/home/node",
                               "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
                               "COPILOT_AUTO_UPDATE": "false", "NO_COLOR": "1"},
                  volumes={str(d / "repo"): {"bind": "/workspace", "mode": "rw"},
                           str(d / "home"): {"bind": "/home/node", "mode": "rw"}})
    elif body.kind == "test":
        t = repo_cfg(meta["repo"]).get("test")
        if not t:
            raise HTTPException(400, "repo has no test command")
        net = t.get("network", "none")
        env = {**(t.get("env") or {})}
        if net == "sandbox":
            env.update({"HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy})
        kw = dict(image=t["image"], command=t["command"], working_dir=t.get("workdir", "/workspace"),
                  user=str(t.get("user", "1000:1000")), environment=env,
                  volumes=_mounts(d, t.get("mounts") or {".": "/workspace"}))
        if net == "sandbox":
            kw["network"] = s.get("network", "agent-sandbox")
        else:
            kw["network_mode"] = "none"
        kw["entrypoint"] = t.get("entrypoint", None)
        if kw["entrypoint"] is None:
            kw.pop("entrypoint")
    else:
        raise HTTPException(400, "kind must be copilot or test")
    dk().containers.run(name=name, labels=labels, detach=True, **_limits(s), **kw)
    save_meta(task_id, meta)
    return {"job_id": name, "kind": body.kind}


def parse_copilot(out: str) -> dict[str, Any]:
    """Final answer, tool calls and usage from `copilot --output-format json` (JSONL)."""
    final, tools, usage = "", [], {}
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("ephemeral"):
            continue
        t, data = ev.get("type", ""), ev.get("data") or {}
        if t == "assistant.message":
            if data.get("content"):
                final = data["content"]
            for tr in data.get("toolRequests") or []:
                tools.append(str(tr.get("name") or tr.get("toolName") or "?"))
        elif t == "session.usage_checkpoint":
            usage = {"premium_requests": data.get("totalPremiumRequests"),
                     "ai_credits": round((data.get("totalNanoAiu") or 0) / 1e9, 3)}
    return {"final": final, "tools": tools[-50:], "usage": usage}


@app.get("/tasks/{task_id}/jobs/{job_id}", dependencies=[Depends(auth)])
def job_status(task_id: str, job_id: str) -> dict[str, Any]:
    d = task_dir(task_id)
    saved = d / "logs" / f"{job_id}.json"
    if saved.exists():
        return json.loads(saved.read_text())
    try:
        c = dk().containers.get(job_id)
    except NotFound:
        raise HTTPException(404, "no such job") from None
    if c.labels.get(f"{LABEL}.task") != task_id:
        raise HTTPException(404, "no such job")
    c.reload()
    if c.status in ("created", "running", "restarting"):
        return {"job_id": job_id, "status": "running"}
    code = int(c.attrs.get("State", {}).get("ExitCode", -1))
    out = c.logs(stdout=True, stderr=False).decode("utf-8", "replace")[-OUT_MAX:]
    err = c.logs(stdout=False, stderr=True).decode("utf-8", "replace")[-20_000:]
    res: dict[str, Any] = {"job_id": job_id, "status": "exited", "exit_code": code, "kind": c.labels.get(f"{LABEL}.kind")}
    if res["kind"] == "copilot":
        res.update(parse_copilot(out))
        res["stderr"] = err[-4000:]
    else:
        res["output"] = ((err + "\n" if err.strip() else "") + out)[-20_000:]   # summary lines last
    (d / "logs" / f"{job_id}.out").write_text(out + "\n--- stderr ---\n" + err)
    saved.write_text(json.dumps(res))
    with contextlib.suppress(Exception):
        c.remove(force=True)
    return res


@app.delete("/tasks/{task_id}/jobs/{job_id}", dependencies=[Depends(auth)])
def kill_job(task_id: str, job_id: str) -> dict[str, Any]:
    task_dir(task_id)
    with contextlib.suppress(NotFound):
        c = dk().containers.get(job_id)
        if c.labels.get(f"{LABEL}.task") == task_id:
            c.remove(force=True)
    return {"ok": True}


def _stage(clone: Path) -> None:
    git(clone, "add", "-A")


@app.get("/tasks/{task_id}/diff", dependencies=[Depends(auth)])
def diff(task_id: str, max_chars: int = 60_000) -> dict[str, Any]:
    meta = load_meta(task_id)
    clone = task_dir(task_id) / "repo"
    _stage(clone)
    stat = git(clone, "diff", "--cached", "--stat", meta["base"])
    files = [f for f in git(clone, "diff", "--cached", "--name-status", meta["base"]).splitlines() if f]
    full = git(clone, "diff", "--cached", meta["base"])
    return {"stat": stat, "files": files, "diff": full[:max_chars], "truncated": len(full) > max_chars,
            "chars": len(full)}


class ApplyBody(BaseModel):
    branch: str
    message: str


@app.post("/tasks/{task_id}/apply", dependencies=[Depends(auth)])
def apply(task_id: str, body: ApplyBody) -> dict[str, Any]:
    meta = load_meta(task_id)
    if not _BRANCH_RE.match(body.branch) or ".." in body.branch:
        raise HTTPException(400, "branch must look like agent/<name>")
    clone = task_dir(task_id) / "repo"
    src = Path(meta["src"])
    if src_git(src, "rev-parse", "--verify", "--quiet", f"refs/heads/{body.branch}", check=False).strip():
        raise HTTPException(409, f"branch {body.branch} already exists")
    _stage(clone)
    if not git(clone, "diff", "--cached", "--name-only", meta["base"]).strip():
        raise HTTPException(409, "no changes to apply")
    # One commit on top of the base, whatever Copilot did with git inside the sandbox.
    git(clone, "reset", "--soft", meta["base"])
    git(clone, "-c", "user.name=agent-harness", "-c", "user.email=agent@localhost",
        "commit", "--quiet", "--no-verify", "-m", body.message)
    commit = git(clone, "rev-parse", "HEAD").strip()
    src_git(src, "fetch", "--quiet", "--no-tags", str(clone), f"HEAD:refs/heads/{body.branch}")
    meta.update(applied=body.branch, commit=commit)
    save_meta(task_id, meta)
    return {"branch": body.branch, "commit": commit, "repo": meta["repo"], "path": meta["src"],
            "base_branch": meta["branch"]}


@app.delete("/tasks/{task_id}", dependencies=[Depends(auth)])
def delete_task(task_id: str) -> dict[str, Any]:
    d = task_dir(task_id)
    for c in dk().containers.list(all=True, filters={"label": [f"{LABEL}.task={task_id}"]}):
        with contextlib.suppress(Exception):
            c.remove(force=True)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


async def reaper() -> None:
    """Kill sandboxes that outlive their timeout; delete task dirs older than task_ttl_h."""
    while True:
        try:
            s = sb()
            now = time.time()
            limits = {"copilot": int(s.get("copilot_timeout_s", 1200)), "test": int(s.get("test_timeout_s", 900))}
            for c in await asyncio.to_thread(dk().containers.list, all=True, filters={"label": [LABEL]}):
                started = int(c.labels.get(f"{LABEL}.started", "0"))
                if now - started > limits.get(c.labels.get(f"{LABEL}.kind", ""), 1200) + 120:
                    log.warning("reaping %s", c.name)
                    await asyncio.to_thread(c.remove, force=True)
            ttl = float(s.get("task_ttl_h", 72)) * 3600
            for d in ROOT.iterdir() if ROOT.exists() else []:
                m = d / "meta.json"
                if d.is_dir() and _TASK_RE.match(d.name) and m.exists() and now - m.stat().st_mtime > ttl:
                    log.info("removing expired task %s", d.name)
                    await asyncio.to_thread(delete_task, d.name)
        except Exception:
            log.exception("reaper failed")
        await asyncio.sleep(300)


@app.on_event("startup")
async def _start() -> None:
    asyncio.create_task(reaper())
