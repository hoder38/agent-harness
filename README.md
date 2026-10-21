# agent-harness (MVP)

MVP of `ARCHITECTURE.md` — phases **1** (compose skeleton, ReAct graph, SSE), **2** (Gate 0, Jev router, decision layer),
**3** (arq queues, cancel, dedupe, kill switch, ZFS backup) and **8** (LINE channel, health alerts).

```
nodeReact nginx ──► agent-api ──► Redis (arq queues, run event streams, killswitch/budget)
LINE webhook   ──►     │  gate0 → gate1(stub) → router(Jev→Qwen→rules) → decide() → enqueue
                       ▼
            agent-worker / agent-worker-bg ──► LiteLLM ──► ollama (qwen3.5:9b) | cloud_fast | cloud_strong
                       │                         └─ egress-proxy (squid allowlist) ─► internet
                       └─► Postgres (LangGraph checkpoints, decisions, usage, alerts) ; Phoenix traces
agent-scheduler: cron alert watcher (containers, queue age, budget) → LINE push
```

## Setup

1. `cp .env.example .env` and fill in the secrets (`host/setup.sh` generates random ones), then add:
   - `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, and set `CLOUD_FAST_MODEL` / `CLOUD_STRONG_MODEL`.
   - `TYPESAFE_API_KEY` for the Jev router. Without it, routing falls back to Qwen and then to rules.
   - `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`, `LINE_OWNER_USER_ID`.
2. `docker compose up -d --build`
3. Pull the local model. The CLI must bypass the proxy to reach its own server:
   ```
   docker compose exec -e OLLAMA_HOST=127.0.0.1:11434 -e NO_PROXY='*' ollama ollama pull qwen3.5:9b
   ```
4. Backups need root once:
   ```
   sudo zfs create tank/agent-backup && sudo chown hoder:hoder /tank/agent-backup
   sudo zfs allow -u $USER snapshot,create,mount,destroy tank/agent-backup
   sudo cp host/systemd/agent-backup.* /etc/systemd/system/ && sudo systemctl enable --now agent-backup.timer
   ```

## Usage

```bash
T=$(docker compose exec -T agent-api python -m app.tools.mint_token | tail -1)
curl -N -H "Authorization: Bearer $T" -H 'Accept: text/event-stream' -H 'Content-Type: application/json' \
  -X POST localhost:8100/api/agent/threads/t1/runs -d '{"text":"what time is it?"}'
```

**Endpoints**
- `POST /api/agent/threads/{tid}/runs` returns JSON, or SSE when you send `Accept: text/event-stream`.
- `GET /api/agent/runs/{id}/stream?token=` streams events for EventSource and supports `Last-Event-ID`.
- `POST /api/agent/runs/{id}/cancel` cancels a run.
- `GET /api/agent/threads/{tid}` returns the thread history.

SSE event types: `decision`, `token`, `tool`, `status`, `done`, `error`, `cancelled`.

**Message prefixes**
- Routing overrides: `/local`, `/cloud`, `/code` (strong model). Private content is detected automatically (PII/secrets/keywords) and stays local.
- Owner commands: `/stop cloud|all`, `/resume`, `/status`.

**Host controls**
- `host/killswitch.sh off|cloud|all`
- `host/gpu-state.sh IDLE|GAMING`
- `host/backup.sh`

**Ports and policy**
- Phoenix UI: http://127.0.0.1:6006. The API is host-only on `127.0.0.1:8100`.
- `config/policy.yaml` and `config/privacy_rules.yaml` hot-reload.
- Test a policy change with: `docker compose exec agent-api python -m app.tools.replay --policy /config/policy.next.yaml --since 14d`

## nodeReact integration (deployed in /home/hoder/nodeReact)

- `config/nginx-release.conf`: `/api/agent/` and `/line/webhook` proxy to `agent-api:8000` (network `anomopi_default`).
- `src/back/controllers/agent-router.js`: `GET /api/agent-token` mints the HS256 JWT (perm=1 admins only;
  optional `AGENT_NODEREACT_USER` narrows it to one username). Needs `AGENT_JWT_SECRET` and `AGENT_OWNER_ID`
  in nodeReact `.env`, both matching this harness's `.env`.
- `src/front/components/Agent.js` + `src/front/agent.js`: the `/Agent` chat page (threads, streaming, Stop, kill switch).
- `integrations/nodereact/` keeps the original reference snippets.

## Tests

```bash
docker build -t agent-harness-api-test -f- api <<'EOF'
FROM python:3.12-slim
COPY requirements*.txt /tmp/
RUN pip install -q -r /tmp/requirements.txt -r /tmp/requirements-dev.txt
EOF
docker run --rm -v $PWD:/app -w /app/api agent-harness-api-test python -m pytest -q
```

## Verified / notes

**Resolved VERIFY items**
- The edge network is `anomopi_default`.
- The Qwen tag is `qwen3.5:9b`. There is no Q6_K tag; override it with `LOCAL_MODEL`.
- The `typesafe-sdk` 0.7.5 API is confirmed.

**Live acceptance checks**
- SSE streaming with tool calls.
- Cancel.
- Thread survives an agent-api restart.
- Squid blocks non-allowlisted domains (403) and agents have no direct egress.
- Kill switch overrides `/cloud`.
- The alert watcher records stopped or unhealthy containers.

**Not yet tested**
- Cloud providers, Jev, LINE delivery and ZFS backup need credentials or root.

**Known limitation**
- agent-api joins `anomopi_default` so nginx can reach it, which gives agent-api direct egress. The workers, which run the tools, stay internal.
