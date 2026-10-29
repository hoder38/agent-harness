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
- `POST /api/agent/runs/{id}/cancel` cancels a run. Cancelling a run that is waiting for your answer also clears the question.
- `POST /api/agent/runs/{id}/resume {"value": "…"}` answers a clarifying question (an option number, free text, or empty to skip). Only the pending run can be resumed; anything else returns 409.
- `GET /api/agent/threads/{tid}` returns the thread history and `pending` (the open clarifying question, if any).
- `GET /api/agent/threads` lists your chats, newest first, built from the decision log. `line` is your LINE thread.
- `DELETE /api/agent/threads/{tid}` hides a chat from the list until its next message. The history is kept.

SSE event types: `decision`, `token`, `reset`, `tool`, `status`, `interrupt`, `done`, `error`, `cancelled`.
- `interrupt` carries `{question, options, round, run_id}`; the run ends in `awaiting_input`.
- `reset` means "discard the streamed text" (a compose retry starts over).
- `done.grounding` = `{verdict, citations, unverified, budget}`.

**Message prefixes**
- Routing overrides: `/local`, `/cloud`, `/code` (coding sandbox, see Coding below). Private content is detected automatically (PII/secrets/keywords) and stays local.
- Owner commands: `/stop cloud|all`, `/resume`, `/status`, `/help`. Any other `/word` replies with the command list.
- `/skip` (web or LINE) skips an open clarifying question; the answer then states its assumptions.

**Web tools** (chat/assistant routes; never on PRIVATE threads)
- `web_search` uses the Brave Search API and needs `BRAVE_API_KEY`; without the key it is not offered.
- `fetch_page` can read any public URL through squid port 4751. Private, LAN, docker and metadata IPs are blocked on both ports. Port 4750 still enforces the allowlist.
- Fetched content is wrapped in `<untrusted>` tags, capped at 2 MB download and 8000 chars of text.

**Grounded answering** (§12a, `grounding.enabled` in `config/policy.yaml`; `false` falls back to the ReAct graph)
- Every answer goes through `app/graphs/grounded.py`. `grounding.mode` picks the path.
  - `cite_only` (default) runs four steps:
    - clarify if the request is ambiguous;
    - decide whether a web search is needed;
    - search, then read up to `cite_pages` (2) pages, preferring official and primary domains and never rejected ones;
    - answer with `[n]` citations to the pages actually read.
  - The cited sources are renumbered 1..k and listed at the end. The only check is that URLs and `[n]` come from pages that were read; there is no per-claim checking and no "Unverified" list. Typical turn: 1–5 s.
  - `verify` runs the full §12a claim check: draft, extract claims, quote-check each claim, then an "⚠️ Unverified" block. It is slower (17–30 s) and noisy on the 9B model.
- Ambiguous messages pause the run with a question (an `interrupt` event on the web; numbered options on LINE). Your next message, or `POST /runs/{id}/resume`, continues the same run. Unanswered questions expire after `grounding.pending_ttl_s`.
- Per-turn caps: 3 searches, 4 page reads, 90 s of active time, $0.05. Hitting a cap ends the answer with a "Verification limit reached" line.
- On local Qwen, every grounded LLM call runs with thinking off (`think: false`). With thinking on, Qwen spends the whole `max_tokens` on hidden reasoning. `grounding.local_think_tasks` can re-enable it per step, but this measured about 140 s per turn, so it is off.

**Coding** (§9a, phase 5): `/code <task>` (name the repo, e.g. "harness" or "前端", or pick it from the question)
- Copilot CLI works in a throwaway clone inside a sandbox container; the harness then runs the repo's tests itself
  (one automatic retry on failure) and asks for approval with the diff. Approve → one commit on a new **local** branch
  `agent/<slug>-<id>` in the original repo; Reject → discarded; any other reply → Copilot revises. Nothing is pushed.
- Needs `COPILOT_GITHUB_TOKEN` (fine-grained PAT with only "Copilot Requests") in `.env`, then
  `docker compose up -d sandbox-manager`. Repos, test commands and Copilot tool permissions: `config/coding.yaml`.
- Build the sandbox image once (and after Copilot upgrades): `docker compose --profile build build sandbox-image`.
- Clones live in `SANDBOX_ROOT` (`/home/hoder/agent-sandboxes`) and are deleted after apply/reject or 72 h.

**Host controls**
- `host/killswitch.sh off|cloud|all`
- `host/gpu-state.sh` shows the GPU state, override and governor heartbeat. `host/gpu-state.sh FREE|BUSY|GAMING` pins a state; `host/gpu-state.sh auto` hands control back to the governor.
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
- `src/front/components/Agent.js` + `src/front/agent.js`: the `/Agent` chat page (threads, streaming, Stop, kill switch). For clarifying questions it shows the question with option buttons, Skip and Cancel, and a typed reply answers it via `/runs/{id}/resume`. It also handles `reset` and shows the source count.
- Deploy front-end changes:
  1. Build the bundle with `docker run --rm -v $PWD/src:/app/src -v $PWD/public:/app/public -w /app reactnode-server:v1.0 ./node_modules/.bin/webpack --config config/webpack.release.config.cjs`.
  2. Rebuild and restart nginx with `docker compose -f docker-compose.release.yml build nginx && docker compose -f docker-compose.release.yml up -d --no-deps nginx`.
- `integrations/nodereact/` keeps the original reference snippets.

## Tests

```bash
docker compose --profile test build agent-test      # after requirements changes
docker compose run --rm agent-test python -m pytest -q -p no:cacheprovider
```
`agent-test` is the `test` stage of `api/Dockerfile`. It mounts `./api` and runs with no network.

## Verified / notes

**Resolved VERIFY items**
- The edge network is `anomopi_default`.
- The Qwen tag is `qwen3.5:9b`. There is no Q6_K tag; override it with `LOCAL_MODEL`.
- The `typesafe-sdk` 0.7.5 API is confirmed.

**Live acceptance checks**
- SSE streaming with tool calls.
- Cancel.
- Thread survives an agent-api restart.
- Squid blocks non-allowlisted domains (403) and agents have no direct egress. The fetch_page port (4751) allows public hosts but returns 403 for private and internal IPs.
- Kill switch overrides `/cloud`.
- The alert watcher records stopped or unhealthy containers.

**Browser** (§12b, `browser` service; read-only)
- Headless Chromium renders a page only when the plain fetch is a JavaScript shell (less than 500 characters of text) or returned 403. This applies to both `fetch_source` and `fetch_page`.
- Each render uses a fresh incognito context. The browser has no host ports and reaches the internet only through squid :4751, so loopback, LAN, internal and metadata URLs get a 403.
- It is configured under `browser:` in `config/policy.yaml`. Measured live: a YouTube channel page took 2.8 s.
- Logging in and interactive browsing are planned as phase 8b-login (§12b).

**Clarify** (§12a): advice, plan and how-to questions now also ask, one per round (up to `clarify.max_rounds` 10), for missing details that would change the answer (goal, level, budget, platform…). Simple factual questions and greetings are answered directly.

**Intent** (§6/§7): Jev also classifies `task_type` (找商品購買 / 知識或建議 / 事實查詢 / 操作教學 / 閒聊) and whether a message corrects the last reply. When the top type is unsure or close to the second (`thresholds.task_confirm_below` / `task_margin`) and shopping is one of them, the bot first asks 「你是想：1. … 2. …」. Answers start with 「我理解為：…」; a correction (「不是，我是說…」) re-classifies the previous + current message and drops the shopping session. Local thinking is on for `compose` only (`grounding.local_think_tasks`).

**Shopping** (§12c, `tools/shop.py`)
- Requests the Jev router classes as `task_type=shopping` (想買…/預算…; 「推薦…」 alone is usually knowledge/advice) ask one question per round, up to 10, about anything you haven't said: item, budget, use, 1–2 key specs, must-haves, brand, store. Each question has option buttons; Skip searches right away.
- It then recommends 3–5 real PChome, momo or Yahoo listings, with price, store, rating and link rendered by code.
- Follow-ups within 30 min refine the search: 便宜一點 lowers the ceiling, and a named brand changes the ranking. Preferences stated with "以後/always" are saved in Redis `shop:prefs`.
- Unrelated listings are filtered by name relevance. Shopee is not used because it serves a CAPTCHA.
- It is configured under `shopping:` in `config/policy.yaml`. Measured live: about 4 s for the clarify step and 7–9 s for recommendations.

**GPU governor** (§11, `gpu-governor` service)
- Polls NVML, `steam-headless` and Ollama every 2 s, and writes `gpu_state` plus a heartbeat to Redis.
- When GAMING starts, it unloads Ollama models from the GPU. Running local jobs older than 20 s move to cloud (or `local_cpu` when private), and private background jobs are deferred.
- The scheduler resumes deferred jobs every 20 s once the GPU leaves GAMING.
- If the heartbeat goes missing, a LINE alert is sent.

**Not yet tested**
- Cloud providers, Jev, LINE delivery and ZFS backup need credentials or root.

**Known limitation**
- agent-api joins `anomopi_default` so nginx can reach it, which gives agent-api direct egress. The workers, which run the tools, stay internal.
