# Agent Harness — Architecture & Implementation Spec

> Audience: an implementing agent/engineer. This document is the source of truth for
> design decisions. Items marked **VERIFY** were not confirmed against current docs and
> must be checked before relying on them.

## 0. Goals

A self-hosted agent harness on a single Linux machine that acts as:

1. **Personal assistant** — chat, todo list, memory, RAG over personal docs, scheduled/background jobs.
2. **Coding agent** — delegates coding to **GitHub Copilot CLI** running in sandbox containers.

Inputs: **nodeReact web app** (complex chat + todos), **LINE bot** (simple chat),
**direct SSH use of Copilot CLI** (must be remembered by the harness), **cron/events**.
Modalities: **text, voice, image, video, documents**.

## 1. Machine & environment constraints

| Item | Value |
|---|---|
| OS | Ubuntu Linux, Docker 29.x + Docker Compose, NVIDIA container runtime |
| CPU / RAM | 16 cores / 46 GB |
| GPU | RTX 5060 Ti, **16 GB VRAM**, no desktop session |
| Disk | ~1.7 TB free |
| Host user | `hoder` (`/home/hoder`) |
| Existing stack (do NOT modify) | `~/nodeReact` compose project: Express 4, React 17, MongoDB 4.4.2, Redis 5, nginx (TLS) |
| Existing GPU user | `~/steam-headless` compose project, container `steam-headless` (game streaming via Moonlight) |
| Sensitive paths (never mount into agent containers) | `~/nodeReact/csr/`, `~/nodeReact/.env*`, `/var/run/docker.sock` (except via socket proxies below) |

**Hard rules**
- Everything runs in **Docker Compose**, project name `agent-harness`, in `~/agent-harness`.
- The harness is **separate** from nodeReact. Integration is via HTTP through nginx only.
- When the GPU is needed for gaming, the harness **must release VRAM** and fall back to cloud/CPU.
- Private data **never** leaves the machine (see §5 Gate 0 and privacy tags).
- Irreversible actions (git push, send mail, delete, money) **always** require human approval.

## 2. Tech choices

| Concern | Choice |
|---|---|
| Orchestration | **Python LangGraph** (library, not LangGraph Server) + **FastAPI** |
| Dev tooling | `langgraph.json` + `langgraph dev` / Studio for local debugging only |
| State / checkpoints | **PostgreSQL 17 + pgvector** (`AsyncPostgresSaver`) |
| Long-term memory | LangGraph `PostgresStore` (with pgvector index) |
| Queue / pubsub / cron / cancel | **Redis 8** + **arq** workers |
| Router model | **TypeSafe Jev** hosted API (`typesafe-sdk`, model `jev-latest`) |
| Local LLM | **Qwen3.5 9B**, Q6_K GGUF via **Ollama** (not vLLM — must unload on demand) |
| Cloud LLMs | **OpenAI** and **Anthropic** APIs directly (own API keys) |
| Coding | **GitHub Copilot CLI** (`copilot`), ACP mode (`--acp`) or headless (`-p`) |
| Embeddings | Qwen3-Embedding-0.6B (or bge-m3), local |
| Reranker | bge-reranker-v2-m3, local |
| Keyword search | `pg_trgm` (or ParadeDB `pg_search` for BM25 + CJK) |
| Doc parsing | Docling |
| ASR | faster-whisper large-v3-turbo or SenseVoice; OpenCC → Traditional Chinese |
| OCR | PaddleOCR (or Docling OCR) |
| Video | ffmpeg (audio extract + scene keyframes) |
| File storage | Plain volume `/srv/agent/files/<sha256[:2]>/<sha256>` + Postgres `attachments` metadata (S3 later only if needed) |
| LLM gateway | **LiteLLM proxy** — single endpoint for Ollama / OpenAI / Anthropic (keys, budgets, retries, fallback, cost log) |
| Observability | **Arize Phoenix** (OTel collector + UI); LangGraph via OpenInference, Copilot CLI via native OTel |
| Egress control | Forward proxy with domain allowlist (e.g. Smokescreen/Squid); all outbound HTTP(S) goes through it |
| Backups | ZFS dataset `tank/agent-backup` on HDD pool `tank` + snapshots (§19) |
| Frontend | Existing nodeReact React 17 app; custom `EventSource` SSE hook (no `useStream`, it needs React ≥ 18) |

**VRAM budget (target ≤ 13 GB steady state)**

| Model | VRAM |
|---|---|
| Qwen3.5 9B Q6_K | ~8 GB |
| KV cache, 32k ctx, q8 (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`) | ~2–3 GB |
| Embedding 0.6B | ~1.2 GB |
| Reranker | ~1.1 GB |
| ASR / VLM | load on demand only, short `keep_alive` |

## 3. High-level flow

```
            nodeReact (chat+todos)   LINE bot (simple chat)   cron / events   SSH copilot (host)
                     │ nginx SSE/REST      │ nginx webhook          │                │ hooks
                     ▼                     ▼                        ▼                ▼
┌ L1 TRIGGER LAYER ─────────────────────────────────────────────────────────┐  ingest only
│ normalize to Job{id, channel, mode, user, payload, attachments, dedupe}   │  (§9)
└──────────────────────────────┬────────────────────────────────────────────┘
                               ▼
┌ L2 INPUT NORMALIZATION (media-worker) ────────────────────────────────────┐
│ text / voice→ASR / image→OCR+VLM / video→ASR+keyframes / doc→Docling      │
│ → {text, attachments[], modality, privacy_tag, summary}                   │
└──────────────────────────────┬────────────────────────────────────────────┘
                               ▼
┌ L3 GATE 0  deterministic (code) ──────────────────────────────────────────┐
│ user override │ secret/PII regex → redact │ source tags → PRIVATE │ budget │
├ L4 GATE 1  Qwen privacy check (local; only when Gate 0 is inconclusive) ──┤
├ L5 GATE 2  Jev router (one call, redacted text only) ─────────────────────┤
│ intent, tier, difficulty, risky, long_ctx, code_multi                     │
├ L6 DECISION LAYER (policy code) ──────────────────────────────────────────┤
│ privacy · confidence · GPU state · channel · mode · budget · fallback     │
│ → Decision{route, model, risky, queue, reasons[]}  (logged)               │
└──────────────────────────────┬────────────────────────────────────────────┘
                               ▼
┌ L7 JOB LAYER  Redis 8 + arq ──────────────────────────────────────────────┐
│ queues: interactive(priority) │ background │ coding │ media │ ingest      │
│ limits, retry, timeout, cancel, dedupe; 1 job = 1 LangGraph thread        │
└──────────────────────────────┬────────────────────────────────────────────┘
                               ▼
┌ L8 EXECUTORS ─────────────────────────────────────────────────────────────┐
│ chat: Qwen / OpenAI / Claude                                              │
│ assistant: LangGraph ReAct + tools (MCP, RAG, todos, memory)              │
│ coding: Copilot CLI in sandbox container (via sandbox-manager)            │
└──────────────────────────────┬────────────────────────────────────────────┘
                               ▼
┌ L9 VERIFY LAYER ──────────────────────────────────────────────────────────┐
│ deterministic checks → model check → escalate tier (max 2) → Inbox        │
│ grounded answering: clarify → claims → 1 primary source each (§12a)       │
├ L10 APPROVAL + OUTPUT ────────────────────────────────────────────────────┤
│ interrupt() for risky actions; Inbox UI; LINE push; SSE; memory write     │
└───────────────────────────────────────────────────────────────────────────┘

Side service: gpu-governor → Redis key gpu_state ∈ {FREE, BUSY, GAMING}
```

## 4. L1 Trigger layer

All inputs become a `Job`:

```python
class Job(BaseModel):
    id: UUID
    channel: Literal["web", "line", "cron", "event", "copilot_ingest"]
    mode: Literal["interactive", "background"]
    user_id: str                    # single owner; reject unknown users
    thread_id: str                  # LangGraph thread
    text: str | None
    attachments: list[AttachmentRef] = []
    override: Literal["local", "cloud", "code"] | None = None
    dedupe_key: str                 # e.g. LINE webhookEventId, mail Message-ID
    created_at: datetime
```

| Channel | Endpoint | Auth | Notes |
|---|---|---|---|
| nodeReact | `POST /api/agent/threads/{id}/runs` (SSE stream), `GET /api/agent/threads/{id}`, `POST /api/agent/runs/{id}/resume`, `POST /api/agent/runs/{id}/cancel`, `/api/agent/todos` CRUD, `/api/agent/inbox`, `POST /api/agent/uploads` | Short-lived JWT issued by nodeReact (see §4a) | Complex chat, todos, approvals, uploads |
| LINE | `POST /line/webhook` | Verify `X-Line-Signature`; allowlist owner LINE userId | Reply quickly with reply token; long jobs reply later via push API. Never send PRIVATE content to LINE. **VERIFY** current reply-token expiry and content-download retention. |
| Cron | arq cron reads `schedules` table | internal | e.g. morning brief, stock check |
| Events | `POST /events/{source}` | per-source secret / HMAC | GitHub webhook, mail poll, price alerts |
| Copilot ingest | `POST /ingest/copilot` on `127.0.0.1:8100` | Bearer token from `~/.config/harness/token` | Bypasses router (§9) |

### 4a. nodeReact → agent-api authentication
- nodeReact (Express) adds `GET /api/agent-token`: for a logged-in owner session, returns a JWT
  signed **HS256** with shared secret `AGENT_JWT_SECRET` (in both `.env` files, never in git):
  `{sub: <owner id>, aud: "agent-harness", iss: "nodereact", iat, exp: now+10min, jti}`.
- Browser sends it as `Authorization: Bearer <jwt>` (fetch) or `?token=` (EventSource; strip from nginx logs);
  refreshes before expiry.
- agent-api verifies signature, `aud`, `iss`, `exp`; `sub` must equal configured owner.
- Change in nodeReact is limited to this one endpoint + the agent chat page; no other nodeReact changes.

Idempotency: `dedupe_key` stored in Redis `SET NX` with TTL 24h; duplicates are dropped.

## 5. L2 Input normalization (media-worker)

1. Sniff real MIME type (not extension); enforce size limits (configurable per channel).
2. sha256 → dedupe; store original in the files volume; insert `attachments` row.
3. Extract:

| Modality | GPU FREE | GPU BUSY/GAMING | Cloud (non-private only) |
|---|---|---|---|
| voice | faster-whisper/SenseVoice on GPU | same on CPU int8 | OpenAI transcription |
| image | OCR + local VLM caption (**VERIFY** Qwen3.5 9B vision support with `ollama show`; else separate VL model loaded on demand) | OCR on CPU; vision → cloud or defer | GPT/Claude vision |
| video | ffmpeg → audio→ASR; scene keyframes (max 16–32) → VLM; build timeline text | CPU, background only | frames + transcript to GPT/Claude |
| doc | Docling → markdown; >~20 pages → chunk + index into RAG | Docling on CPU | Claude/GPT for hard layouts |

4. Strip EXIF/GPS before any cloud upload.
5. Output: `{text, attachments[], modality, privacy_tag, summary}`.
6. Video and large docs are always `mode=background`; the user gets an immediate "processing" reply.

## 6. L3–L5 Gates

### Gate 0 — deterministic (pure code, must be < 5 ms)
- Honour user override (`/local`, `/cloud`, `/code`).
- Owner commands are handled before Gate 0 (`pipeline.handle_command`), never reach a model, and reply in English:
  `/stop cloud|all`, `/resume`, `/status`, `/help`. Any other leading `/word` (e.g. `/new`) replies "Unknown command" + the
  `/help` list. Paths such as `/etc/hosts …` are not commands.
- Secret/PII regex scan (API keys, tokens, private keys, ID numbers, card numbers, phone, email). Redact for any non-local use.
- **Source tags:** content originating from mail, calendar, finance docs, `~/private`, personal notes ⇒ `privacy_tag=PRIVATE`.
- Budget check (daily $ per provider, Copilot AI credits).

### Gate 1 — Qwen privacy check (DEFERRED — not in initial build)
Ship without Gate 1. Add it only if the decision log shows Gate 0 missing private data.
Keep the interface (`gate1(text) -> is_private`) as a no-op stub returning `None`.
- Only when Gate 0 is inconclusive on free-form text. Output: `is_private: bool`.
- When `gpu_state=GAMING`: skip and treat inconclusive as PRIVATE for background jobs.

### Gate 2 — Jev router
Single `system_one` call on **redacted** text/summary only. Never send PRIVATE content.
If PRIVATE, use a Qwen-based fallback router with the same output schema.
If PRIVATE **and** `gpu_state=GAMING` (Qwen unloaded): use the **rule-based router** — no model call:
`/code` or repo/path/stack-trace patterns → `coding` (deferred, sandbox is fine but cloud is not → background, wait for FREE);
todo/reminder/calendar keywords → `assistant`; otherwise → `chat`; tier `local_cpu`; confidence fixed at 0.6.

```python
questions = {
  "intent": Choice(instructions="What the user wants",
     criteria={"coding": "Write, debug, review code or repos",
               "assistant": "Todos, calendar, email, notes, reminders, search, documents",
               "chat": "General question answerable directly"}),
  "tier": Choice(instructions="Least capable model that can do this well",
     criteria={"local": "Simple chat, summarise, rewrite, classify, short lookups",
               "cloud_fast": "Moderate reasoning, multi-step tool use",
               "cloud_strong": "Hard reasoning, multi-file code, architecture, unknown failures"}),
  "difficulty": Score(instructions="Task complexity",
     criteria=["Trivial", "Moderate", "Hard multi-step"]),
  "risky": Noul(instructions="Implies destructive, external, or money-related action"),
  "long_ctx": Noul(instructions="Requires reading many files or very long documents"),
  "code_multi": Noul(instructions="Requires changing code across multiple files"),
  "ambiguous": Noul(instructions="Refers to something by a pronoun or vague word (that, it, 那個, 它) "
                                 "whose meaning is not given in the message or the previous message"),
  "task_type": Choice(instructions="What kind of answer the user wants for the current message",
     criteria=TASK_TYPES),  # shopping | advice | factual | howto | chitchat (jev_router.py)
  "correction": Noul(instructions="The current message says the reply to the previous message misunderstood it, "
                                  "or restates what the user actually wanted (不是…, 我是說…, I meant…)"),
}
```
**Task type and corrections (2026-10-29).** `task_type` picks the graph branch (only `shopping` searches stores);
`RouterResult.task_probs` keeps Jev's probability per type. The Qwen/cloud router returns `task_type` +
`task_confidence`; the rule router uses explicit purchase words (想買/購買/預算/500以下, buy) → shopping, how-to words
→ howto, else advice. A `correction` > `thresholds.correction` (0.6) with a previous message makes the pipeline route
`previous + "\n" + current` again as one fresh request (the first result is discarded) and pass that text to the graph
as `correction`; the graph answers it and drops the shopping session. Measured on Jev: real corrections 0.95,
other follow-ups ≤ 0.1; 「推薦適合冷凍且富含纖維素的食材」 → advice 1.0, 「想買調味料罐 500以下」 → shopping 0.98.
Router input is `[Previous message] … \n[Current message] …` when the thread had a message within
`clarify.previous_ttl_s` (Redis `prevmsg:{thread_id}`), so follow-ups are not judged ambiguous on their own.
`ambiguous` wording was calibrated on Jev (2026-10-22): broader "missing time/place/version" phrasing scored most
ordinary questions > 0.7; the referent-only wording scores vague messages ~0.95 and clear ones ≤ 0.55.
The Qwen/cloud JSON router returns the same field; the rule router sets 0.
**VERIFY** exact SDK class names (sync/async client) and response field names in `typesafe-sdk`.
On Jev error/timeout (> 2 s): fall back to Qwen router; if GAMING, fall back to a cheap cloud model router (non-private only).

## 7. L6 Decision layer

Pure, unit-tested function. Policy values in `config/policy.yaml` (hot-reloadable).

```python
def decide(job, gate0, router, gpu_state, budget) -> Decision:
    # order matters; first matching rule wins for each field
    # 1. override
    # 2. PRIVATE  → model must be local
    #       if gpu_state == GAMING: routing = rule-based (§6, no model call);
    #       execution: interactive → local_cpu (warn, short answers); background → defer until FREE
    # 3. intent confidence < 0.8 OR router.ambiguous > 0.7 → route="clarify" on a LOCAL model (local_cpu if GPU BUSY/GAMING);
    #    no tier bump — never pay a cloud model to ask a question. /cloud override and PRIVATE keep their model.
    #    The answer is re-routed together with the original request + question (§12a MVP), so the tier
    #    (and any escalation) is decided once the intent is clear. After 5 clarify rounds → answer with assumptions.
    # 4. code_multi > 0.6 or long_ctx > 0.6 → cloud_strong (cloud_fast if over budget)
    # 5. gpu_state in (BUSY, GAMING) and model == local and not PRIVATE → cloud_fast
    # 6. channel == "line" → prefer local/cloud_fast, short answer, no coding approvals inline
    # 7. mode == "background" → stricter budget, risky actions queued for approval
    # 8. budget exceeded → downgrade + notify
    # 9. video / large doc → queue="background"
    # 10. risky > 0.5 → risky=True (forces approval before tool execution)
```

```python
class Decision(BaseModel):
    route: Literal["chat", "assistant", "coding", "clarify", "defer"]
    model: Literal["local", "local_cpu", "cloud_fast", "cloud_strong"]
    queue: Literal["interactive", "background", "coding", "media"]
    risky: bool
    reasons: list[str]          # human-readable rule trail, shown in UI and logged
```

Model aliases are defined in **LiteLLM** `config/litellm.yaml`; the harness only calls
`http://litellm:4000/v1` with `model=<alias>`:
```yaml
model_list:
  - model_name: local
    litellm_params: { model: ollama_chat/qwen3.5:9b-q6_K, api_base: http://ollama:11434 }   # VERIFY tag
  - model_name: local_cpu
    litellm_params: { model: ollama_chat/qwen3.5:9b-q4_K_M, api_base: http://ollama:11434, num_gpu: 0 }
  - model_name: cloud_fast
    litellm_params: { model: openai/<fill: fast model>, api_key: os.environ/OPENAI_API_KEY }
  - model_name: cloud_strong
    litellm_params: { model: anthropic/<fill: strong model>, api_key: os.environ/ANTHROPIC_API_KEY }
router_settings:
  fallbacks: [{ cloud_strong: [cloud_fast] }, { cloud_fast: [cloud_strong] }]   # never fall back to/from local here; privacy is decided upstream
litellm_settings: { max_budget: <daily USD>, budget_duration: 1d }
```
Privacy rule: LiteLLM fallbacks must never route a `local` request to a cloud model.
The model is fixed for the duration of one agent loop; escalation starts a new loop.

Every decision is written to `decisions` table (job_id, inputs, jev raw output, decision, latency, cost) for later threshold tuning.

## 8. L7 Job layer

- Redis 8 + arq. Queues: `interactive` (highest priority), `background`, `coding`, `media`, `ingest`.
- Concurrency: GPU semaphore (Redis) — max 2 concurrent local LLM calls; `coding` max 2 sandboxes; `media` max 2.
- Each job runs a LangGraph graph with `thread_id`; checkpointer = `AsyncPostgresSaver`.
- Cancel: Redis key `cancel:{run_id}` checked between graph steps.
- Retry with exponential backoff on transient provider errors; timeout per queue.
- Events streamed via Redis pub/sub `run:{run_id}` → agent-api → SSE to browser.

## 9. Copilot CLI integration

### 9a. Harness → Copilot (coding route)
- `sandbox-manager` creates a throwaway container per coding task:
  - image with `copilot` CLI, git, language toolchains
  - mounts only a **git worktree** of the target repo
  - CPU/mem limits, no GPU, egress allowlist (GitHub, Copilot endpoints, package registries)
  - env `COPILOT_GITHUB_TOKEN` (hidden via `--secret-env-vars`) — use the **least-privileged token type Copilot CLI accepts**
    (VERIFY: fine-grained PAT with only "Copilot Requests" permission); repo push uses a **separate** token that only
    the harness uses *after* approval, never given to Copilot
  - env `HTTPS_PROXY=http://egress-proxy:4750`. **Phase-5 gate:** confirm Copilot CLI honours `HTTPS_PROXY`
    (run with proxy logging; all Copilot traffic must appear in proxy log). If it does not, give the sandbox
    a dedicated network with a firewall allowlist (iptables/nftables in sandbox-manager) instead.
- Preferred: **ACP** (`copilot --acp`) so permission requests map to LangGraph `interrupt()` approval cards.
- Headless alternative:
  ```bash
  copilot -p "$TASK" -C /workspace --output-format json --stream on -s \
    --model "$MODEL" --max-ai-credits 20 \
    --allow-tool 'write' --allow-tool 'shell(npm test)' \
    --deny-tool 'shell(git push)' --deny-tool 'shell(rm)' \
    --secret-env-vars=COPILOT_GITHUB_TOKEN
  ```
  **VERIFY** tool-pattern syntax with `copilot --help`. Never use `--allow-all`.
- After Copilot finishes: harness runs tests/lint itself (verify layer), shows diff, `interrupt()` before commit/push/PR.
- Sandbox Copilot sessions use a per-sandbox `COPILOT_HOME`; continuation across sessions uses stored summaries, not `--resume` across homes.

### Implemented 2026-10-28 (phase 5) — headless, one approval at the end
**Flow** (`api/app/graphs/coding.py`, route `coding` = `/code …`; no LLM of our own — the request goes to Copilot verbatim):
1. `start`: refuse on PRIVATE threads and when the kill switch is `cloud|all` (Copilot is a cloud service). Pick the repo
   from `config/coding.yaml` (`repos.<name>.aliases`, whole words); if none or several match, ask with a clarify
   interrupt whose options are the repo names.
2. `run_copilot`: sandbox-manager clones the repo and runs `copilot -p` in a throwaway container; progress lines stream
   as `token` events.
3. `run_tests`: the harness runs the repo's own test command in a **second** container (no token, `network: none`).
   On failure Copilot gets **one** retry (`coding.max_retries`) with the test output tail.
4. `review` → `approve`: a clarify interrupt (`kind: approval`) with Copilot's summary, test result, diffstat and the
   first `diff_chars` of the diff; options Approve / Reject (核准 / 拒絕). Any other reply = a revision request
   (Copilot re-runs with it, up to `max_revisions` 2). Works on web and LINE unchanged; the question stays answerable
   for `approval_ttl_s` (3 days). `/skip` = reject.
5. `apply`: squash everything into one commit and `git fetch <clone> HEAD:refs/heads/agent/<slug>-<id>` into the
   original repo. The working tree and current branch of the original repo are not touched; **nothing is pushed**
   (no push token exists anywhere). The task dir is then deleted.

**Deviations from the plan above**
- **Clone instead of worktree**: `git clone --local --no-hardlinks` into `SANDBOX_ROOT/<task>/repo` (a worktree would
  write into the original repo's `.git` while Copilot works). Uncommitted changes in the original are not included.
- **Headless instead of ACP**: no per-tool approval cards; permissions are fixed allow/deny lists in
  `config/coding.yaml` (`write`, `shell(git:*)`, read-only shell tools, `python3`, `node`; denied: `git push`,
  `git remote`, `git config`, `rm`, `curl`, `wget`). Deny beats allow. The container is the real boundary.
- **Sandbox root** is `/home/hoder/agent-sandboxes` (`SANDBOX_ROOT` in `.env`; no root on the host for `/srv/agent`).
- Flags used: `copilot -p … -s --output-format json --no-ask-user --no-auto-update --disable-builtin-mcps
  --secret-env-vars=COPILOT_GITHUB_TOKEN --max-ai-credits 2000 --allow-tool … --deny-tool …`. The final answer is the
  last `assistant.message` event; usage comes from `session.usage_checkpoint` (`totalPremiumRequests`,
  `totalNanoAiu`/1e9 recorded as `ai_credits`) and is stored in the run outcome (`done.coding.usage`).

**sandbox-manager** (`sandbox-manager/app.py`, FastAPI on `agent` + `dockerctl`, bearer = `HARNESS_TOKEN`):
`POST /tasks` (clone) · `POST /tasks/{id}/jobs {kind: copilot|test}` · `GET /tasks/{id}/jobs/{job}` ·
`DELETE …/jobs/{job}` · `GET /tasks/{id}/diff` · `POST /tasks/{id}/apply {branch, message}` · `DELETE /tasks/{id}`.
- It mounts only `<repo>/.git` of each configured repo (no working tree, no `.env`), plus `SANDBOX_ROOT` at the same
  path (so sandbox bind mounts resolve on the host). Read-only rootfs, `cap_drop: ALL`.
- Sandbox containers: uid 1000, `cap_drop: ALL`, `no-new-privileges`, 4 CPU / 6 GB / 1024 pids, `/tmp` tmpfs, at most
  `max_running` 2 at once. Copilot containers join only the internal `agent-sandbox` network (squid :4750 is the
  only reachable host); test containers default to `network: none`. Mounts: the clone at `/workspace` and the task's
  own home (`COPILOT_HOME`, `trustedFolders: [/workspace]`).
- A reaper kills containers older than their timeout (+2 min) and deletes task dirs older than `task_ttl_h` (72 h).
- Cancel: the graph polls `state.is_cancelled` while a job runs and kills the container (also on task abort).

**Test commands** (`repos.<name>.test`): agent-harness runs pytest in `agent-harness-api:test` with the clone's
`api/` and `config/` mounted; nodeReact runs jest in its own `reactnode-server:v1.0` image (node 14 + `node_modules`
symlinked from `/app`), skipping the integration suites and `shioaji-tool` (they need Mongo/Redis).
Measured on the unmodified repos: agent-harness 158 passed; nodeReact 3840 + 1326 passed (~4 min).

**Verified live 2026-10-28**: `/code` → repo question → answer resumes the coding graph; a missing token ends the run
with a clear error. From `agent-sandbox`, only `egress-proxy:4750` is reachable (direct internet: ENETUNREACH; Redis,
Postgres, sandbox-manager and socket-proxy-rw do not resolve); squid allows `api.github.com` and denies `example.com`.
**End-to-end run (2026-10-28)**: `/code` README edit on agent-harness → Copilot edit → harness tests passed → approval
question with diff (21 s) → "1" → commit on a local `agent/…` branch, sandbox cleaned up; 1 premium request
(6.8 AI credits). **Phase-5 gate passed:** all Copilot traffic went through squid (`api.github.com`,
`api.business.githubcopilot.com`, `telemetry.business.githubcopilot.com`; `cafe.github.com` is denied and not needed).
The PAT must be user-owned with the account permission "Copilot Requests"; otherwise Copilot fails with
"Authentication failed" even though `/user` works.

### 9b. SSH Copilot → harness memory (ingest)
Host-level Copilot hooks (`~/.copilot/hooks/harness.json`):
```json
{ "version": 1, "hooks": {
  "agentStop":  [{ "type": "command", "command": "~/.copilot/hooks/harness.sh", "timeoutSec": 5 }],
  "sessionEnd": [{ "type": "command", "command": "~/.copilot/hooks/harness.sh", "timeoutSec": 5 }]
}}
```
```bash
#!/bin/sh
# ~/.copilot/hooks/harness.sh — forward hook payload; never block Copilot
curl -s -m 3 -X POST http://127.0.0.1:8100/ingest/copilot \
  -H "Authorization: Bearer $(cat ~/.config/harness/token)" \
  -H 'Content-Type: application/json' --data-binary @- >/dev/null 2>&1 || true
```
Payload contains `sessionId`, `cwd`, `timestamp`, and (`agentStop`) `transcriptPath`, (`sessionEnd`) `reason`.

Ingest worker (mounts `/home/hoder/.copilot/session-state:/copilot-sessions:ro`):
1. Parse `<sessionId>/events.jsonl` (JSONL events: `session.start`, messages, tool calls, …) incrementally (store last processed event id).
2. Extract: user prompts, final answers, files changed, commits, PRs, repo/branch.
3. Redact secrets → summarize with **local Qwen** (defer if GAMING).
4. Write: LangGraph thread `copilot:<sessionId>`; `PostgresStore` namespace `("me","coding_episodes")` with embedding; extracted facts → `("me","profile")`.
5. Nightly reconciler cron ingests sessions with no recorded `sessionEnd`.

### Implemented 2026-10-30 (phase 6) — hooks, ingest job, `harness-memory` MCP
- **Hooks:** `host/copilot-hooks/harness.{json,sh}`, installed by `host/install-copilot.sh`. The script adds the hook name as `event` to the payload and posts to `POST /ingest/copilot` (`app/routes/ingest.py`, harness token); it always exits 0.
- **Debounce:** `agentStop` fires after every agent turn, so those share one job deferred by `copilot_ingest.debounce_s` (60 s; the job re-reads the whole file). `sessionEnd` queues immediately and records `ended` in Redis.
- **Job** `ingest_copilot` (`app/jobs/copilot_jobs.py`, queue `ingest`, same worker as `background`): parse → redact → summarise with the local model (waits while GAMING or kill switch `all`) → write. Skipped when `last_event_id` is unchanged. The profile/procedure extractor runs once, on session end, over the owner's prompts only.
- **Writes:** thread `copilot:<sessionId>` (redacted prompts and final answers, in the LangGraph checkpointer, not listed in the web thread list); `coding_episodes/<sessionId>` (summary + repo, files, commits, PRs); extracted facts in `profile`/`procedures`. Every ingest is audit-logged (kind `memory`, keys only).
- **Reconciler:** `reconcile_cron` at 03:30 queues sessions idle ≥ `reconcile_idle_s` and modified within `reconcile_days` that have no recorded `sessionEnd`.
- **MCP:** `app/mcp_server.py`, streamable HTTP at `http://127.0.0.1:8100/mcp`, harness token required, anything else on that mount is 404. Tools: `recall`, `remember`, `get_thread_summary`. Copilot is a cloud model, so PRIVATE memories are never returned. Also `list_todos`/`add_todo` (phase 7).
- **Not verified live:** the hook payload field names follow the GitHub hooks reference (`sessionId`, `cwd`, `reason`); run one real SSH session after `host/install-copilot.sh` and check `docker compose logs agent-worker-bg`.

### 9c. Harness memory → SSH Copilot (recall)
Run a `harness-memory` **MCP server** (inside agent-api, bound to 127.0.0.1) and register it in host Copilot CLI MCP config. Tools: `recall(query)`, `remember(fact)`, `list_todos()`, `add_todo(title, due?)`, `get_thread_summary(thread_id)`.

## 10. Memory, RAG, todos

- **Short-term:** LangGraph checkpoints per thread; summarisation node when context > threshold.
- **Long-term (`PostgresStore`) namespaces:** `("me","profile")`, `("me","episodes")`, `("me","coding_episodes")`, `("me","procedures")`.
- **RAG tables:** `documents`, `chunks(id, doc_id, text, embedding vector, tsv/trgm, source, path, date, privacy_tag, hash)`; HNSW index.
- Chunking: 400–800 tokens, ~15% overlap, heading-aware. Re-index on hash change only.
- Query: hybrid (vector + keyword) top-50 → RRF → local reranker → top-5 with citations.
- RAG is exposed as a **tool** (`search_knowledge(query, filters)`), not auto-injected.
- Coding agents do not use vector RAG (Copilot greps the repo).
- **Todos:** single table `todos(id, title, notes, status, due_at, source_channel, created_at, updated_at)` — used by nodeReact UI (REST), agents (tools), Copilot (MCP).
- Retrieval eval set: 30–50 real questions with expected sources; track recall@5.

### Implemented 2026-10-29 (phase 10)

**Knowledge base (`api/app/rag/`)**
- Corpus: a read-only docs folder (`DOCS_DIR`, default `~/agent-docs`, mounted at `/docs`) plus uploads via `POST /api/agent/kb/documents` (stored in Postgres only). The vault in §10a will be a second root.
- `index_docs_cron` rescans every `rag.index_interval_min` (5) minutes under a Redis lock. A file is re-indexed only when its hash changes. The hash covers the text, chunk settings, `chunking.VERSION` and embed model. Embeddings are reused per (model, title, chunk hash). Files that disappear are removed, but nothing is removed when the root itself is missing.
- Chunking is heading-aware, ≤ 800 tokens with 15% paragraph overlap. Each section is chunked on its own so a citation's heading is exact, which means many chunks are under 400 tokens. A tiny intro is merged only into its own following subsection, never into a sibling.
- Embeddings: `qwen3-embedding:0.6b` via Ollama (1024 dims, HNSW). Queries get the instruct prefix. While GAMING the model is loaded with `num_gpu: 0`.
- Search:
  - vector top-50 + keyword top-50 (Postgres `simple` tsvector; CJK indexed as character bigrams), fused with RRF;
  - then the top 10 go to the CPU cross-encoder `reranker` service (`bge-reranker-v2-m3`), then the top 5, at most one chunk per (path, heading);
  - a hit is kept if its rerank score ≥ 0.01 **or** its vector similarity ≥ 0.45;
  - if the reranker is down or times out, results stay in RRF order with the vector floor.
- Privacy:
  - A document is PRIVATE when its frontmatter says `privacy: private`, it sits in a `private/` folder, it contains a private keyword, or it contains a high-confidence secret pattern.
  - Contact details and `api_key: os.environ/...` config examples do not make a document PRIVATE.
  - PRIVATE chunks are returned only for private runs, which use local models.
  - On non-private searches, hits are redacted with the Gate 0 rules.
- **Measured recall@5** on `config/rag_eval.yaml` (33 zh/en questions about this repo's docs; `python -m app.tools.rag_eval --both`, results stored in `rag_eval_runs`):

  | Configuration | recall@5 | MRR | ms/query |
  |---|---|---|---|
  | Rerank (10 candidates) | **1.000** | 0.872 | ~4400 |
  | RRF only (reranker down) | 0.879 | 0.828 | ~7 |

  Settings sweep:
  - 20 rerank candidates gave no recall gain and took ~8 s.
  - Using the rerank score alone as a floor (≥ 0.05–0.2) dropped correct hits: cross-language paraphrases score near 0, so recall fell to 0.73–0.85.
  - Off-topic queries scored ≤ 0.004 on rerank and ≤ 0.25 on vector similarity, so the either/or floor keeps them out.

**Long-term memory (`api/app/memory.py`, LangGraph `AsyncPostgresStore` with a pgvector index on `text`)**
- Namespaces: `("me", profile | procedures | episodes | coding_episodes)`.
- Each item stores `{text, privacy, pinned, kind, source, history}`.
- Writes:
  - Explicit `/remember <fact>` (web/LINE; the fact is private if the thread or fact is private, or Gate 0 finds anything) and the `remember` tool.
  - **Automatic extraction:** after each non-coding turn, an `extract_memory` job on the background queue asks the local Qwen for add/update/delete ops. Update/delete may touch only the memories the extractor was shown.
  - **Episodes:** a thread idle for 30 minutes (with ≥ 2 turns) is summarised into `episodes` (key = thread id; expires after 180 days unless pinned; `sweep_memory_ttl` runs hourly).
  - Background jobs defer while GAMING or when the kill switch is `all`.
- Conflicts: the newest text wins and the previous 5 versions are kept in `history`.
- Recall:
  - **Grounded `cite_only` graph:** memories (score ≥ 0.5, pinned first) and KB hits are fetched once per question, *before* the clarifying step. The clarifier is told what is already known, so it doesn't ask for it. A KB hit with rerank ≥ `rag.skip_clarify_score` (0.5) skips the question, even when the router flagged the request ambiguous, because the router cannot see the documents.
  - Memories go into the compose prompt as notes "that may be outdated". KB hits become sources `[n] 📄 title — path#heading` before any web sources. If they answer the question, no web search runs.
  - Private threads get KB and memories but no web.
  - ReAct chat/assistant also get the `search_knowledge`, `recall_memory` and `remember` tools.
- Coding runs neither recall nor extract.
- Agent/extraction/Copilot/MCP memory writes also appear as low-priority Inbox items (phase 7); owner-explicit `/remember` and Memory-page edits do not. Every write (`/remember`, tool, extraction, UI edit/delete) is audit-logged (kind `memory`, keys only) and visible on the Memory page.

## 10a. Knowledge bank — local Obsidian vault

The vault is plain Markdown on the host and is the **human-curated source of truth** for
knowledge. `PostgresStore` remains the agent's own memory; RAG indexes the vault.

**Location & access**
- Vault: `/home/hoder/vault` (host directory, git-tracked).
- Machine is headless → edit from other devices via **Syncthing** (container) to Obsidian
  desktop/mobile, and optionally a browser-accessible Obsidian container
  (`lscr.io/linuxserver/obsidian`, **VERIFY** image), bound to LAN/VPN only.
- Alternative sync: Obsidian Sync (paid) or Self-hosted LiveSync (CouchDB).

**Folder convention**
```
vault/
  Inbox/          # quick captures (LINE, web) written by the harness
  Notes/          # human notes
  Notion/         # imported from Notion (read-mostly)
  Projects/  Areas/  Resources/  Archive/
  AI/             # harness-owned: Daily/ YYYY-MM-DD.md summaries, Copilot session notes
  Attachments/
```
- Agents may **write only** to `Inbox/` and `AI/` without approval; edits elsewhere require `interrupt()` approval and are shown as a diff.
- Privacy via frontmatter `privacy: private|normal` + folder rules in `config/privacy_rules.yaml` → `privacy_tag` on every chunk.

**Indexing (vault-indexer, part of agent-worker)**
- inotify/watchdog on the vault (mounted **read-only** into the indexer) → debounce → hash check → re-index changed files only.
- Parse frontmatter (tags, aliases, privacy), headings, `[[wikilinks]]`, `![[embeds]]`.
- Store `note_links(src, dst)` → retrieval can expand to 1-hop linked notes.
- Attachments (PDF, images) → media-worker (Docling/OCR) → indexed with the parent note.
- Chunk heading-aware; citations are `[[Note#Heading]]` so answers link back to Obsidian.

**Writing (vault-writer tool, separate mount rw on `Inbox/` and `AI/` only)**
- Tools: `capture_note(text, tags)`, `append_daily(summary)`, `propose_edit(path, diff)` (approval).
- Nightly cron: daily summary of chats, todos done, and Copilot sessions → `AI/Daily/`.
- Nightly cron: `git add -A && git commit` in the vault for history/rollback.

**Copilot (SSH)**: expose `search_notes(query)` / `read_note(path)` via the `harness-memory` MCP server, or start Copilot with `--add-dir ~/vault` for direct read access.

**Notion migration (one-off job)**
1. Notion → Settings → Export all workspace content → **HTML**, include subpages and files (split by teamspace if large).
2. Obsidian **Importer** plugin (official) → Notion → import zip into `vault/Notion/` (**VERIFY** whether current Importer also supports Notion API import and how databases map to Obsidian Bases).
3. Post-process script: add frontmatter `source: notion`, `notion_id`, `privacy`; fix remaining links; convert database CSVs to notes or Bases; move files into `Attachments/`.
4. Validate: page count vs Notion, broken-link report, spot-check databases and embeds.
5. Index into RAG; keep Notion read-only for ~1 month, then archive.

## 11. GPU governor

Container with NVML access (`capabilities: [utility]`) and read-only Docker events via socket proxy.

| State | Enter when | Exit when | Action |
|---|---|---|---|
| FREE | — | — | normal |
| BUSY | GPU util > 70% for 30 s or free VRAM < 8 GB | below thresholds for 2 min | new non-private local jobs → cloud_fast |
| GAMING | `steam-headless` container running **and** (NVENC encoder util > 0 or GPU util > 30%) | 5 min after condition clears | unload Ollama models (`POST /api/generate {"model": …, "keep_alive": 0}`), embeddings/reranker/ASR → CPU, all non-private → cloud, private background → defer |

- Publishes `gpu_state` and transitions to Redis (`SET gpu_state`, `PUBLISH gpu_state_changed`).
- In-flight local jobs: short ones finish; long ones are requeued on cloud (non-private) or paused.
- On return to FREE: resume deferred jobs.

### Implemented 2026-10-28 (phase 4)
- **Where:**
  - `app/governor.py`, run as compose service `gpu-governor` (same image as the api, `python -m app.governor`).
  - The NVML device reservation is `capabilities: [utility]`.
  - Thresholds live in `policy.yaml gpu_governor`.
- **Polling, not Docker events:** every `poll_s` (2 s) it reads:
  - NVML (util, encoder util, VRAM);
  - `steam-headless` via `socket-proxy-ro` (`containers/json` name filter);
  - Ollama `/api/ps`;
  - `sem:gpu` (the harness's own GPU jobs).
- **Transition rules (`Governor.step()`, pure and unit-tested):**
  - While the harness holds the GPU, its own load does not count as BUSY util.
  - Free VRAM is computed as total − used + Ollama's own resident models, so a loaded Qwen does not trigger BUSY by itself.
  - Entering GAMING unloads only Ollama models with `size_vram > 0`.
- **Redis keys:**
  - `SET gpu_state` and `PUBLISH gpu_state_changed`;
  - `LPUSH gpu_governor:transitions` (last 200);
  - `gpu_governor:heartbeat` (TTL `heartbeat_ttl_s`).
  - `host/gpu-state.sh FREE|BUSY|GAMING` sets `gpu_state_override`, which wins; `auto` deletes it.
- **Alerts:** if the heartbeat is missing or reports an error, the alert watcher sends a LINE alert (`governor:heartbeat` / `governor:error`). `gpu_governor.enabled: false` turns this off.
- **Jobs before start:**
  - When GAMING and the decision is `local`:
    - non-private jobs go to `cloud_fast` (`local_cpu` if the kill switch is on);
    - private interactive jobs go to `local_cpu`;
    - private background jobs are deferred.
  - Deferred jobs are stored in the Redis hash `deferred`.
  - The scheduler cron `resume_deferred` (every 20 s) re-enqueues them once the state is not GAMING.
- **Jobs in flight:**
  - A `local` run older than `inflight_short_s` (20 s) when GAMING starts is stopped.
  - It is then requeued (status `requeued`) on the alternative model and continues from its checkpoint, or deferred if it is private background work.
  - Shorter runs finish.

## 12. L9 Verify layer & L10 Approval/Output

> **Status:** only coding verify (phase 5) and grounded claim verification (phase 8a) are implemented. The general model check + tier escalation below is planned as phase 10b (§15); until then no run is escalated or sent to the Inbox for failing verification.

Verify (skip for trivial chat):
1. Deterministic: coding → harness-run tests/lint/build; assistant → schema valid, no tool errors, citations present when RAG used.
2. Model check: Jev `Noul("The response fully and correctly answers the request")` for non-private; Qwen for private; deterministic-only when GAMING + private.
3. Fail → escalate tier (local → cloud_fast → cloud_strong), max 2 escalations → Inbox item.

Approval:
- `interrupt()` before: git push/PR merge, send mail/message to others, delete, calendar create/delete, any money action, network enablement in sandbox, any action flagged `risky`.
- Background approvals persist (Postgres checkpoint) → Inbox card in nodeReact + LINE push notification → resume on approval.

Output: SSE (web), LINE reply/push (never PRIVATE content), Inbox, memory write, PR link.

## 12a. Grounded answering (clarify → verify claims → cite) — decided 2026-10-22, implemented 2026-10-28 (phase 8a; default mode cite_only)

Owner preferences, applied to **every** answer:
1. **Clarify first** whenever intent/scope is uncertain. *Intent* uncertainty → ask the user; *factual* uncertainty → search (never ask "should I search?").
2. **One authoritative, primary source that directly supports each factual claim.** Secondary sources are never cited.
3. **Insufficient evidence → partial answer** with unverified content explicitly labelled.

Claims are classified first: greetings, creative writing, opinion, pure reasoning/math are `exempt` (no source); every factual claim needs a citation or an "unverified" label.
PRIVATE threads never use web tools; they may cite only the local knowledge base (§10a). Until that ships (phase 10/10a), every factual claim on a PRIVATE thread is labelled unverified.

### MVP subset (implemented 2026-10-22)
The full graph below is phase 8a. The MVP ReAct graph already does:
- **Clarify on local, escalate after:** decision rule 3 (§7) sends ambiguous requests (intent confidence < `clarify_below` 0.8, or Jev `ambiguous` > `thresholds.ambiguous` 0.7 — intent confidence alone misses vague referents like 「那個東西多少錢」, which scored 0.98) to `route=clarify` on `local` (`local_cpu` when the GPU is BUSY/GAMING). Redis `clarify:{thread_id}` (`{text, redacted_text, question, rounds}`, TTL `clarify.context_ttl_s`) makes the next message route as `request + [Assistant asked] … + [User answered] …`; the router's tier on that text decides any escalation. On PRIVATE threads the question is left out of the redacted copy. `clarify.max_rounds` (10, raised from 5 on 2026-10-28) → answer with stated assumptions. `clarify_rounds` is logged in `decisions.inputs` for replay.
- **Confirm the task type (rule 3b, 2026-10-29):** when the top `task_type` probability < `thresholds.task_confirm_below` (0.7) or leads the second by < `thresholds.task_margin` (0.3), and one of the two is in `clarify.confirm_task_types` (`[shopping]`: only types that change the pipeline), `Decision.confirm_task` holds the two types and the grounded graph asks the owner to pick before anything else. Not asked on resumes, PRIVATE threads, non-chat routes or after `clarify.max_rounds`.
- **Prompt rules:** `GROUNDING_WEB` (one primary source read with `fetch_page`, `[n]` citations, "(unverified)"/「（未驗證）」 labels, exempt kinds) or `GROUNDING_OFFLINE` (no invented sources; one "⚠️ 未驗證" line) on every non-clarify route; the clarify hint asks one question with ≤ 3 options.
- **Tool caps:** `policy.yaml tool_limits` (`web_search: 3`, `fetch_page: 4`) per user turn; calls over the cap get an error `ToolMessage` without running.

Deferred from the MVP (land with phase 8a):
- **Citation URL check** — after the run, every URL in the answer must be one actually read by `fetch_page` in this run; otherwise append a warning line. Implement as a final graph node that rewrites the last `AIMessage` (same id), so the checkpointed thread, the `done` event, and LINE delivery all carry the warning.
- **Source tier labels** — `config/authority.yaml` (see Authority tiers below) applied in `web_search`/`fetch_page` output as `tier="primary|official:<entity>|secondary|reject|unknown"` so the model can pick sources; rules only, no LLM judge. Deferred because the domain list needs upkeep and adds little without per-claim verification.

### Integration
- New graph `app/graphs/grounded.py` replaces the ReAct graph in `worker.run_agent`; the ReAct graph stays for Studio/regression.
- `decision.route == "clarify"` becomes a hard trigger (`force_clarify=True`), not just a prompt hint.
- New structured functions `search_web()` / `fetch_source()` are called **by nodes, not by LLM tool-calling**, so budgets are enforceable. The string tools in `tools/web.py` remain for the ReAct graph.
- Implement `POST /runs/{run_id}/resume` (currently 501).

### State schema (`app/graphs/grounded_state.py`)
```python
ClaimKind   = Literal["factual", "greeting", "creative", "opinion", "reasoning", "math", "instruction"]
ClaimStatus = Literal["pending", "verified", "unverified", "contradicted", "exempt"]
Tier        = Literal["T1_primary", "T2_secondary", "T3_reject", "unknown"]

class Clarification(TypedDict): round: int; question: str; answer: str | None
                                slot: str   # referent|scope_time|scope_region|version|goal|conflict
class Claim(TypedDict):
    id: str; text: str; kind: ClaimKind; status: ClaimStatus
    entity: str | None           # subject of the claim (decides which site is "official")
    time_sensitive: bool
    source_id: str | None        # exactly one
    quote: str | None            # verbatim, ≤ 300 chars
    reason: str | None           # shown to the user when unverified/contradicted
class Candidate(TypedDict): url: str; title: str; snippet: str; domain: str; age: str | None
                            tier: Tier; tier_reason: str; for_claims: list[str]
class Source(TypedDict): id: str; url: str; final_url: str; domain: str; title: str
                         tier: Tier; tier_reason: str; fetched_at: str; published_at: str | None
                         text_sha256: str; excerpt: str   # relevant windows only, ≤ 8000 chars
class Budget(TypedDict): search_used: int; fetch_used: int; clarify_used: int
                         usd: float; active_ms: int      # waiting for the user is not counted
                         exhausted: str | None           # search|fetch|clarify|usd|time
class Answer(TypedDict): text: str; verdict: Literal["complete", "partial", "exempt_only"]
                         citations: list[dict]; unverified: list[str]; assumptions: list[str]

class GroundedState(MessagesState):
    run_id: str; channel: Literal["web", "line"]; private: bool; model: str; force_clarify: bool
    question: str                                     # redacted turn text + clarification answers
    clarifications: list[Clarification]; assumptions: list[str]
    draft: str                                        # internal, never streamed
    claims: Annotated[list[Claim], merge_by_id]
    candidates: Annotated[list[Candidate], merge_by_id]   # keyed by url
    sources: dict[str, Source]; queries: list[str]
    budget: Budget; turn_started_ms: int
    answer: Answer | None; check_failures: list[str]
```
`ingest` resets all per-turn fields on each new user turn; `messages` keeps accumulating. Sources from earlier turns are stubbed in model input (like `stub_old_tool_results`).

### Node flow
```
START → ingest → assess_ambiguity ──ambiguous & clarify_used<10─▶ ask_user (interrupt) ──resume──┐
                      │◀──────────────────────────────────────────────────────────────────────────┘
                      ▼ clear, or cap reached → assumptions
                    draft → extract_claims ──no factual claims──────────────────────────────▶ compose
                                  ▼
                 ┌─▶ plan_queries → search → classify_candidates → fetch → verify_support ─┐
                 └──────────── pending factual claims & budget left ◀──────────────────────┘
                                  ▼ all resolved | budget exhausted
                    compose → final_check ──fail (1st)──▶ compose
                                  ▼ pass | 2nd fail → downgrade to unverified
                    interpretation_conflict? ──yes & clarify_used<5──▶ ask_user
                                  ▼
                                 END
```
A `budget_guard` conditional edge sits before every costly node and routes to `compose` when a limit is hit. The research loop runs at most 3 times (= search cap), so it always terminates.

| Node | Kind | Behaviour |
|---|---|---|
| `ingest` | code | reset per-turn state; `private/channel/model` injected from `Decision` |
| `assess_ambiguity` | LLM, structured, T=0 | → `{ambiguous, slot, question_to_user, options[≤3], assumption_if_skipped}`. Ask if: unclear referent; missing time/region/version that changes the answer; ≥2 plausible readings with different answers; conflicting constraints; `force_clarify`; and (since 2026-10-28) for advice/plans/recommendations/how-to, any missing detail that would change or personalise the answer (goal, audience, level, budget, platform, constraints). Most important first, one question per round, re-assessed after each answer; never for greetings or simple one-answer facts. A question matching an earlier one in the turn ends clarifying (no loops) |
| `ask_user` | `interrupt()` | `clarify_used += 1`; resume value → `clarifications[-1].answer`, merged into `question`. At the cap, `assumption_if_skipped` → `assumptions` |
| `draft` | LLM | internal draft (`tags=["internal"]`; the worker does not emit its tokens) |
| `extract_claims` | LLM + code override | atomic claims + kind. Code forces `factual` for sentences with numbers, dates, prices, versions, or named entity + factual verb |
| `plan_queries` | LLM | ≤ remaining searches; one query may cover several claims; `site:` when the entity's official domain is known |
| `search` | tool | §Tools; `search_used += 1` |
| `classify_candidates` | rules → LLM for `unknown` | §Authority; `T3_reject` is never fetched |
| `fetch` | tool | best tier × claim coverage first, ≤ remaining fetches; re-classify by `final_url` after redirects |
| `verify_support` | LLM quote + code checks | §Direct support; a claim is bound to one source once verified |
| `compose` | LLM, streamed | §Output; 20 % of USD and 15 s are reserved for it |
| `final_check` | code | §Final check |
| `interpretation_conflict` | code + LLM | primary sources disagree on what the question means (e.g. two definitions) → ask, or list both readings as assumptions |

A claim backed only by T2 sources → `unverified` ("only secondary sources found").

### Tools (structured, called by nodes)
```python
async def search_web(query: str, *, count=5, freshness: Literal["pd","pw","pm","py"] | None = None,
                     ledger_key: str) -> SearchResult      # {query, hits[{url,title,snippet,domain,age}], error, cached}
async def fetch_source(url: str, *, focus: list[str], max_chars=8000,
                       ledger_key: str) -> FetchResult     # {url, final_url, status, content_type, title,
                                                           #  published_at, excerpt, text_sha256, truncated, error, cached}
```
- Reuse Brave endpoint, strict egress proxy, `check_url`, `fetch_proxy`, `MAX_DOWNLOAD_BYTES`, `html_to_text`. Queries are checked against Gate 0 rules before sending.
- **Idempotency ledger:** Redis `ledger:{thread}:{turn}:{sha1(args)}` (TTL 1 d); arq `Retry` / checkpoint replays hit the ledger and are not charged again.
- **Fetch cache:** `fetch:{sha1(final_url)}` 24 h; cache hits are free.
- **Focus windows:** ±600 chars around matches of claim keywords/numbers, merged up to `max_chars` (avoids the fixed head truncation cutting the evidence).
- HTTP 429/errors still consume one unit (no retry loops). PDFs are unsupported for now → claim unverified ("primary source is a PDF"); Docling later.
- `published_at` from `article:published_time`, `<time datetime>`, or `Last-Modified`.
- PRIVATE: `kb_search()` / `kb_read()` with the same interfaces (`domain = vault:<path>`); until §10a ships they return `error="local KB not available"`. (Phase 10: in the default `cite_only` mode, private threads already cite the owner's KB from §10, with no web; verify mode below is unchanged.)

### Authority tiers (`config/authority.yaml`) — rules first, LLM only for edge cases
```yaml
primary_suffixes: [".gov", ".gov.tw", ".gov.uk", ".europa.eu", ".edu"]   # .edu only for the institution itself
primary_domains:  [rfc-editor.org, ietf.org, w3.org, iso.org, law.moj.gov.tw, sec.gov, mops.twse.com.tw, doi.org, arxiv.org]
entity_domains:   { OpenAI: [openai.com, platform.openai.com], Anthropic: [anthropic.com, docs.anthropic.com] }
secondary_domains: [reuters.com, apnews.com, cna.com.tw, wikipedia.org]
reject_patterns:  [medium.com, reddit.com, quora.com, stackoverflow.com, "*.blogspot.com"]
```
Order: reject → T3; entity's own domain → T1 (only for claims **about that entity**); primary suffix/domain and the publisher is the originator → T1; secondary → T2 (Wikipedia is used only to find the primary it cites); GitHub → T1 only for the entity's official org; otherwise `unknown` → LLM judge that sees **only URL, title, snippet, claim** (no page body; limits injection), cached by `(domain, entity)` for 30 days.

### Direct support (all must hold for `verified`)
1. Source tier is `T1_primary`.
2. The LLM-extracted `quote` is a substring of `excerpt` after NFKC/whitespace/width normalisation (prevents fabricated quotes).
3. Every number, date, and version in the claim appears in the quote (`1,000` == `1000`).
4. Entailment judge (T=0) says the quote alone `supports` the claim; `refutes` → `contradicted` and the answer states what the source says.
5. Time-sensitive claims need `published_at`, or the answer notes "per page fetched on {date}".
6. Several qualifying sources → pick one: entity's own domain > newer > shorter excerpt.

### Final check (code)
- Each verified claim maps to exactly one `[n]`; each `[n]` is a T1 entry in `sources`.
- No URL in the answer that is not in `sources`.
- Each unverified/contradicted claim is in the "⚠️ Unverified" block or ends with "(unverified)".
- Exempt content has no citations; non-empty `assumptions` → an "Assumptions" paragraph.
- 1st failure → recompose with the failure list; 2nd failure → deterministic downgrade (uncited sentences move to the unverified block).

### Output
```
<answer; verified sentences end with [1]>

Assumptions: …                      (if any)

⚠️ Unverified:
- <claim> (reason: only secondary sources / source is a PDF / verification budget exhausted / source says …)

Sources:
[1] <title> — <url> (fetched 2026-10-22)
```
LINE uses the same structure without markdown; PRIVATE threads still send `private_line_notice`. Labels follow the user's language (Traditional Chinese for Chinese).

### Interrupt / checkpointer
- `AsyncPostgresSaver`, `thread_id = job.thread_id`; every node checkpoints; budget counters live in state and the ledger lives in Redis, so crashes/retries resume without double charging.
- `ask_user`: `interrupt({"type": "clarify", "round", "question", "options", "private"})`.
- Worker: `astream(..., stream_mode=["messages", "updates"])`; on `__interrupt__` → emit SSE `interrupt`, `runmeta.status = awaiting_input`, Redis `pending:{thread_id} = {run_id, round, created_at}` (TTL 24 h). LINE replies with the question (PRIVATE → `private_line_notice`). Waiting time is not counted in `active_ms`.
- Pipeline: after commands, if `pending:{thread_id}` exists, the next message (web or LINE) is the **answer**; Gate 0 still runs, then a resume job (new `run_id`, `parent_run_id`) runs `astream(Command(resume=text), cfg)`. `/skip` (or an empty web resume) resumes with the sentinel `"/skip"`, because `Command(resume=None)` counts as "no resume" and re-raises the interrupt. The graph then answers using `assumption_if_skipped`.
- Pending expired → the next message starts a new turn (clear the pending task first if needed; see §17).
- Web: `POST /runs/{run_id}/resume {value}`; `useAgentThread` renders the question + option buttons on `interrupt`. Cancel deletes `pending:{thread_id}`.

### Cost limits (`policy.yaml`, per user turn across resumes)
```yaml
tool_limits: { web_search: 3, fetch_page: 4 }   # already used by the MVP graph
clarify: { max_rounds: 10, context_ttl_s: 86400 } # already used by the MVP decision layer
grounding:
  enabled: true
  per_turn: { max_usd: 0.05, max_active_s: 90 }  # search/fetch/clarify caps come from the keys above
  reserve_for_compose: { usd_fraction: 0.2, seconds: 15 }
  brave_usd_per_query: 0.0          # VERIFY plan price
  llm_authority_cache_days: 30
  max_claims: 8                     # factual claims checked per turn; the rest are labelled unverified
  pending_ttl_s: 86400              # an unanswered clarifying question expires after this
  local_think_tasks: []             # Qwen thinking per step on local; off (see Implementation notes)
  local_think_tokens: 2048
executor:
  recursion_limit: 60               # 5 clarify rounds ×2 + 3 research loops ×5 + fixed nodes ≈ 35
```
- Before each LLM call: `est = in_tokens_est·price_in + max_tokens·price_out` (`prices_usd_per_mtok`); skip to `compose` if `usd + est > max_usd·(1 − reserve)`. Actual usage is added afterwards.
- Each node checks `active_ms + elapsed > (max_active_s − reserve.seconds)·1000`.
- Local models cost $0 and are bounded only by time and counts. Daily provider budgets, kill switch, and GPU semaphore still apply first.
- Exhaustion → `budget.exhausted` is set and the answer ends with "⚠️ Verification limit reached ({reason}); some content is unverified."

### Light mode `grounding.mode: cite_only` (default since 2026-10-28)
The owner found per-claim verification too heavy: 17–30 s per turn and long "Unverified" lists on the 9B model. Preference 2 (one primary source per claim) and preference 3 (an explicit unverified label) are relaxed to "cite the pages you read".
- **Flow:** `ingest → assess_ambiguity ⇄ ask_user/wait_answer → research → compose → final_check`.
- **research:**
  - One LLM call returns `{needs_web, entity, queries ≤ 2}`.
  - Even if the model says no, the turn searches anyway when `_FACT_HINT` matches.
  - It then searches, ranks hits by authority tier (rules only; rejected domains are dropped) and fetches up to `cite_pages` pages of `cite_chars` each.
  - PRIVATE threads do no web access.
- **compose:** answers with `[n]` after sentences taken from a source, and may use general knowledge without a number.
- **final_check:** URL and `[n]` validity only (recompose once, then strip).
- **Output:**
  - Cited sources are renumbered 1..k in order of first use and listed at the end.
  - The verdict is `cited`, `uncited` or `no_sources`.
- **Unchanged:** clarify, resume, `/skip`, caps and streaming.
- **Live timing:** 1–5 s per turn.
- `mode: verify` keeps the full graph described above.

### Implementation notes (2026-10-28)
Deviations from, and details beyond, the design above:
- **Graph shape and history:**
  - `interpretation_conflict` runs before `compose`.
  - State lists are replaced wholesale (no reducers).
  - LLM history is plain user/assistant text (`plain_history`; no tool messages).
- **Sources and privacy:**
  - Only T1 candidates are fetched.
  - On PRIVATE threads, every factual claim is labelled unverified (`private_kb`) until §10a ships.
- **Streaming:**
  - Only `compose` tokens are streamed. A retry emits SSE `reset` first.
  - Exempt-only answers (e.g. greetings) arrive in `done` only.
  - `done.grounding` = `{verdict, citations, unverified, budget}`.
- **Clarify replies:**
  - A digit reply maps to that option's text.
  - An empty reply or `/skip` sets `skip_clarify`.
  - The reply message id is `reply-{run_id}-{n}`.
- **Resume:** the pipeline checks `pending:{thread}` before commands. On resume:
  - decision rule 3 (clarify) is skipped;
  - `decisions.inputs.resume_of` is logged;
  - the parent run's status becomes `answered`.
- **Search tries:** each claim is searched at most twice (`MAX_TRIES`); queries already run in this turn are not repeated.
- **Numeric check:**
  - Month names next to a digit count as numbers, so "October 2, 2023", "Oct. 2, 2023" and 「2023年10月2日」 match each other.
  - Dotted versions (e.g. `3.12.0`) are kept as one number.
- **Local Qwen (qwen3.5:9b):**
  - Every grounded call sends `think: false`, because with thinking on the model spends all of `max_tokens` on hidden reasoning and returns empty content.
  - `grounding.local_think_tasks` re-enables thinking per step with `local_think_tokens` extra headroom, but [verify_support, compose] measured about 140 s per turn (over `max_active_s`) with empty answers, so it is `[]`.
  - The draft and compose prompts state the answer language explicitly; the 9B model drifted to Chinese on "the user's language".
  - Claim rows are given to compose as `(true) … [n]` / `(false) … source [n] says` / `(unchecked) …`, so the model doesn't copy "VERIFIED" labels into the answer.
- **Measured live (2026-10-28, local GPU):**
  - simple factual turn: 17–30 s;
  - clarify interrupt: about 1 s.
  - Verification quality is bounded by the 9B model. Wrong drafts are caught (labelled "source says …" or corrected), but a supported claim is sometimes left unverified.
- **Tests:** 35 grounded tests in `test_grounded.py` and `test_grounded_worker.py`. They use a task-dispatching fake chat model instead of `GenericFakeChatModel`, and run via `docker compose run --rm agent-test`.

### Tests (pytest + `GenericFakeChatModel` + `InMemorySaver` + `fakeredis`; monkeypatch `search_web`/`fetch_source`)
| Area | Cases |
|---|---|
| Clarify | ambiguous referent → one interrupt with `slot=referent`; clear question → no interrupt; `route=clarify` always asks; 5 ambiguous rounds → no 6th interrupt, answer has assumptions; web resume merges answer into `question`; LINE next message resumes, Gate 0 runs, `/skip` works; PRIVATE clarify sends notice only on LINE; waiting 10 min (fake clock) not counted in `active_ms` |
| Authority | openai.com is T1 for OpenAI pricing, not for Anthropic pricing; only reuters/wikipedia → unverified and not cited; medium.com never fetched; redirect to rejected domain → rejected; unknown-domain LLM verdict cached (2nd call skips LLM) |
| Support | quote not in excerpt → not verified; claim "$20" vs quote "$25" → `contradicted` with source's value; two T1 sources → one `[n]`; time-sensitive w/o date → fetched-date note; prompt injection in page does not change tiers or output |
| Partial | 1 of 3 claims verified → `partial` with 2 reasons; greeting/creative only → no search, `exempt_only`; poem + one fact → only the fact needs a source; PRIVATE (KB missing) → no web calls, all facts unverified |
| Budget | 5 queries needed → stops at 3, `exhausted=search`; 6 fetches → 4, cache hits free; USD estimate over 80 % → skip research but still compose, total ≤ $0.05; active time 76 s → compose, ≤ 90 s; arq `Retry` replay → ledger hit, no double count; Brave 429 → one unit, no loop |
| Check / regression | URL not in `sources` → repair, then downgrade; existing `test_graph.py` / `test_pipeline_worker.py` (cancel, kill switch, LINE delivery, `repair_tool_calls`) pass; checkpoint with 3 sources < 100 KB |

## 12b. Browser — decided 2026-10-28 (8b-read implemented 2026-10-28; 8b-login planned)

The owner wants the agent to read JavaScript-heavy pages now, and later to log in to sites and act on them. These are two stages with very different risk.

### Stage 1 — read-only rendering (implemented 2026-10-28)
- **Service:** `browser` (`browser/`, image pinned to `mcr.microsoft.com/playwright/python:v1.64.1-noble`). It is a FastAPI wrapper around headless Chromium.
  - `POST /render {url, timeout_s, settle_ms}` → `{final_url, status, title, html}`; `GET /healthz`.
  - Each request gets one fresh incognito context: no cookies, storage, service workers or downloads survive it. Images, media and fonts are skipped. At most 2 renders run at once.
- **Isolation:**
  - It sits on the `agent` (internal) network only and has no host ports.
  - Chromium's only route out is the squid open port `egress-proxy:4751`, which refuses private, LAN, metadata and Docker addresses. `--proxy-bypass-list=<-loopback>` forces even `127.0.0.1` through squid.
  - The container is `read_only`, with a tmpfs `/tmp`, `cap_drop: ALL`, `no-new-privileges`, `mem_limit` 1.5 GB and `pids_limit`.
  - Verified live: loopback, internal services, 192.168.x and 169.254.169.254 all get a squid 403, which the harness discards.
- **Use (fallback only):** `tools/browser.py`, called from `fetch_source` (grounded graph) and `fetch_page` (ReAct).
  - The browser is used only after a plain fetch, when that fetch returned HTML with fewer than `browser.min_text_chars` characters of text (a JS shell) or a status in `browser.render_on_status` (`[403]`).
  - The rendered text replaces the plain one only if it is longer, and is still subject to `check_url`, `<untrusted>` wrapping and the fetch cache. The result carries `rendered: true`.
  - If the renderer is down or fails, the plain result stands.
- **Cost:** about 0.5–3 s per rendered page, and 0 for normal pages. It runs on the CPU, so the GPU governor is unaffected.
- **Policy:** `browser: {enabled, min_text_chars: 500, render_on_status: [403], timeout_s: 20, settle_ms: 1500}`.

### Stage 2 — interactive browsing and login (phase 8b-login, planned; not built)
Goal: the agent can navigate, click, fill forms and use sites where the owner is logged in (for example check an order, read a dashboard, download a statement). Design decisions:
- **Separate service `browser-session`.** It is not the stage-1 renderer, so read-only research can never touch a logged-in profile.
  - It also sits on the internal network only and uses squid, but with **its own per-site allowlist** (`config/browser_sites.yaml`: domain → profile, allowed actions). Navigation outside the listed domains is blocked in the browser (route abort) and in squid (a separate port with a dstdomain ACL).
- **Profiles and credentials:**
  - One Playwright persistent context (`storage_state`) per site. It is stored encrypted (age/Fernet, key in `.env`) in a Docker volume, never in Postgres or Redis and never in prompts or logs.
  - The owner logs in **once, by hand**, through a noVNC view of the session browser (host-only port or nginx behind nodeReact auth). The agent never sees or types passwords.
  - 2FA and CAPTCHA are always handed back to the owner via the same view, with a LINE or web notification.
  - There is no password manager integration in the first cut.
- **Tools (structured, node-called, like §12a):**
  - `open(site, path)`, `snapshot()` (accessibility tree + text, capped; the screenshot goes to the owner only), `click(ref)`, `type(ref, text)`, `select(ref, value)`, `back()`, `download()`.
  - Element refs come from the snapshot (Playwright aria refs), not free-form selectors chosen by the LLM.
- **Approvals (§16 taint rules apply; all page content is untrusted):**
  - **Read actions** (navigate within the site, open details, paginate) are auto-allowed, within a per-run cap on steps and seconds.
  - **Write actions** (submit a form, buy, send, delete, change a setting, any POST that is not a search) need `interrupt()` approval. The owner sees a summary and a screenshot in the `/Agent` page and LINE, then approves or rejects.
  - Sites can be marked `read_only: true` in `browser_sites.yaml`, which forbids write actions entirely.
  - Money and credential pages (checkout, password, payment settings) are blocked unless explicitly allowed per site.
- **PRIVATE:** a logged-in session's content is treated as PRIVATE (local model only, no cloud) unless the site entry says otherwise. Downloads land in a quarantine dir and go to the media pipeline (phase 9).
- **Audit:** every action is logged with URL, action, ref, approval id and a screenshot hash in Postgres. A per-site kill switch is part of the global kill switch (§19).
- **Acceptance:**
  - Owner logs in via noVNC once and the session survives restarts.
  - The agent reads an order list on an allowlisted site.
  - Any submit pauses for approval.
  - Navigation to a non-listed domain is blocked.
  - A cloud model never receives logged-in page text unless the site is marked non-private.

## 12c. Shopping — clarify intent and budget, recommend real listings — implemented 2026-10-28 (phase 8c)

**Goal:** "幫我推薦調味料罐" should end with 3–5 real products the owner can buy. Each one shows its real price, store, rating and link. Before that, the bot asks for whatever is missing (item and budget). It is a branch of the grounded graph (§12a), so the web clarify UI and LINE resume work unchanged.

- **Flow:** `ingest → [confirm_intent → ask_user ⇄ wait_answer] → shop_slots → (ask_user ⇄ wait_answer → shop_slots) → shop_search → shop_compose → shop_check → END`. Non-shopping turns go to `assess_ambiguity` as before. Private threads never shop.
- **Trigger (2026-10-29):** the router's `task_type == "shopping"` (§6), not keywords or the slot model; 「推薦…」 alone is usually `advice`. When decision rule 3b flags the task type as unsure, `confirm_intent` asks 「你是想：1. 找商品購買 2. 知識或建議」 (options from `TASK_LABELS`, no LLM); a number or label picks the type, `/skip` keeps the router's pick, free text uses the router's reading of the exchange. A follow-up within `session_ttl_s` (30 min) that the router still classes as shopping, such as "cheaper" or "another brand", refines the last search.
- **Echo line:** every grounded answer except chitchat starts with 「我理解為：<label>—「<request>」」 (code-built, also streamed first), so a misread is visible and can be corrected (§6 correction).
- **Slots (one LLM call, JSON):** item, item_specific, search_terms (1–2 store keywords), budget_min/max, no_budget, use_case, brand_pref, must_have, avoid, remember. Code enforces the following, because a 9B model gets these wrong:
  - Budget and brand words (知名品牌/有品牌) are stripped from the search terms. Budgets become a price filter and brand preference becomes ranking.
  - An explicit budget in the user's own lines (`parse_budget`: 1000以下, 300到600, 大概1000, 2千以上, under $500) overrides the model's reading. The newest clarify answer wins.
  - `item_specific=false` is ignored when a search term of 3+ characters appears verbatim in the user's words, so "電動胡椒研磨器" is never asked about again.
  - `no_budget` only counts if the user said 不限/隨便/any. Otherwise the bot asks.
  - Clarify asks one question per round, in a fixed order, up to `max_clarify_rounds` (10):
    1. item, then budget. Each is asked at most twice, and the question is always about only that slot.
    2. use case (who it's for / where it's used)
    3. 1–2 key specs the model picks for the item type (spec:容量, spec:尺寸, spec:材質…)
    4. must-have / must-avoid features
    5. brand (skipped if a preference is saved)
    6. store (buttons: PChome / momo / Yahoo / 不限). The choice limits which stores are searched.
  - Anything the user already said or answered (including 不限) is not asked. The model's per-slot questions are carried across rounds within the turn. A question too similar to one already asked is skipped (≥ 50 % shared bigrams). If the model forgets to fill a slot, the user's answer fills it.
  - When the model gives no question for a slot, a default question is used (except for spec). Once a budget is known, a question that mentions price falls back to that default. Options appear as buttons, never in the question text. Skip searches immediately with what is known.
  - A turn that started shopping stays shopping through its clarify answers (「要降噪」 alone is not a shopping request). Later-turn refinements ("cheaper") do not ask again.
  - A refine keeps the previous slots and never re-asks for an item that is already known. "Cheaper" (便宜一點/cheaper) always lowers the ceiling to below what was shown (`budget_max`×0.7) and clears the floor.
  - Standing preferences ("以後都這樣", "always") go to Redis `shop:prefs`. If the model leaves `remember` empty, the stated brand_pref/avoid is saved anyway. Saved preferences apply to later searches.
- **Stores (`tools/shop.py`):** each adapter is a URL builder plus a parser. All of them search server-side with a price filter and need no login:

  | Store | Source | Price filter |
  |---|---|---|
  | PChome 24h | JSON API `ecshweb.pchome.com.tw/search/v4.3` | `&price=lo-hi` |
  | momo | schema.org ItemList (JSON-LD) in the search page | `&_advPriceS/E` |
  | Yahoo購物中心 | schema.org ItemList (JSON-LD) | `&minp/&maxp` |

  - Shopee is excluded because it redirects bots to a CAPTCHA (§12b bot wall).
  - feebee and BigGo expose only aggregate data. They could be added later as price-comparison sources.
  - Fetches go through squid. Results are cached in Redis for `cache_s`. Out-of-stock and out-of-range listings are dropped, duplicates are merged, and stores are interleaved round-robin.
- **Relevance (code):** stores pad results with loosely related goods ("電動" → 沖牙器, 起子機).
  - Bare modifiers are dropped from search terms (a term needs at least 2 character bigrams or words).
  - A listing is kept only if its name contains at least `min_relevance` (0.5) of a term's bigrams. Results are sorted by relevance, then by brand preference.
  - The top `max_candidates` (15) go to the model.
- **Real listings only:**
  - The model sees numbered listings and writes only `[n] why it fits` lines plus one tip.
  - Code renders each pick's name, NT$price, store, ★rating and URL from the adapter data.
  - Unknown numbers and non-listing URLs are dropped.
  - The answer ends with the search criteria and "prices as of <time>, check the product page".
  - `shop_check` recomposes once if no valid pick was made. With no listings, the bot says so and suggests adjustments; it never names products.
  - `answer.products` holds the picks; they also fill `citations` for the UI source count.
- **Measured live (qwen3.5:9b, web):** about 3–4 s to ask for the budget, then 6–9 s to recommend.
  - Full clarify, live: "想買保溫杯" asked budget → use → capacity → brand → store (5 rounds, about 4–10 s each), then gave 5 matching 480–500 ml cups at NT$400–600. "推薦藍牙耳機 3000以下 送禮用" did not ask about use (送禮 was already given). It asked about features → fit → battery → brand → store, then picked in-ear ANC models.
  - Multi-turn test: "想買電動胡椒研磨器" → budget question → "1000 左右" gave 4 grinders at NT$879–959. "便宜一點" then gave 5 grinders at ≤ NT$600. "我偏好有品牌的，以後都這樣" then gave branded picks.
- **Policy:** `shopping:` in `config/policy.yaml` (enabled, stores, max_clarify_rounds, max_candidates, min_relevance, picks, per_store, cache_s, session_ttl_s).
- **Tests:** `api/tests/test_shopping.py`:
  - parser, budget, stock and duplicate filtering;
  - ask budget → only real listings rendered, invented numbers and URLs dropped;
  - refine, and "cheaper" overriding a model that raises the floor;
  - relevance filter;
  - no_budget guard;
  - one detail per round in order, reworded duplicates skipped, Skip stops asking, mid-clarify answers stay in shopping;
  - standing-preference fallback;
  - budget parser, and a model that drops the budget or brand;
  - non-shopping turns, private threads.
- **Later:** product cards in `/Agent` (image, price, button) from `answer.products`; feebee/BigGo price comparison; price-drop watch as a scheduled job; LINE flex carousel.

## 13. Docker Compose (skeleton)

```yaml
name: agent-harness
services:
  agent-api:
    build: ./api
    ports: ["127.0.0.1:8100:8000"]     # host-only (Copilot hooks, MCP)
    env_file: .env
    depends_on: [agent-postgres, agent-redis]
    networks: [agent, edge]
  agent-worker:
    build: ./api
    command: arq worker.Settings
    volumes: ["/home/hoder/.copilot/session-state:/copilot-sessions:ro"]
    networks: [agent]
  agent-scheduler:
    build: ./api
    command: arq worker.CronSettings
    networks: [agent]
  media-worker:
    build: ./media                       # ffmpeg, faster-whisper, Docling, PaddleOCR, opencc
    command: arq media.Settings
    volumes: [hf-cache:/root/.cache/huggingface]
    deploy:
      resources:
        limits: { cpus: "8", memory: 12g }
        reservations: { devices: [{ driver: nvidia, count: all, capabilities: [gpu] }] }
    networks: [agent]
  gpu-governor:
    build: ./governor
    environment: [DOCKER_HOST=tcp://socket-proxy-ro:2375]
    deploy: { resources: { reservations: { devices: [{ driver: nvidia, count: all, capabilities: [utility] }] } } }
    networks: [agent]
  ollama:
    image: ollama/ollama
    environment:
      - OLLAMA_KEEP_ALIVE=10m
      - OLLAMA_MAX_LOADED_MODELS=2
      - OLLAMA_FLASH_ATTENTION=1
      - OLLAMA_KV_CACHE_TYPE=q8_0
    volumes: [ollama:/root/.ollama]
    deploy: { resources: { reservations: { devices: [{ driver: nvidia, count: all, capabilities: [gpu] }] } } }
    networks: [agent]
  agent-postgres:
    image: pgvector/pgvector:pg17
    volumes: [pg:/var/lib/postgresql/data]
    networks: [agent]
  agent-redis:
    image: redis:8
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}", "--appendonly", "yes"]
    volumes: [redis:/data]
    networks: [agent]
  litellm:
    image: ghcr.io/berriai/litellm:main-stable          # VERIFY tag
    command: ["--config", "/app/config.yaml"]
    volumes: ["./config/litellm.yaml:/app/config.yaml:ro"]
    environment: [HTTPS_PROXY=http://egress-proxy:4750, NO_PROXY=ollama]
    env_file: .env                                       # OPENAI_API_KEY, ANTHROPIC_API_KEY
    networks: [agent]
  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["127.0.0.1:6006:6006"]                       # UI host-only; OTLP on 4317/6006
    volumes: [phoenix:/mnt/data]
    environment: [PHOENIX_WORKING_DIR=/mnt/data]
    networks: [agent]
  egress-proxy:                                          # only service on the egress network
    image: ghcr.io/stripe/smokescreen                     # VERIFY image; alt: squid with ACL allowlist
    volumes: ["./config/egress-allowlist.yaml:/etc/smokescreen/acl.yaml:ro"]
    networks: [agent, egress]
  sandbox-manager:                       # implemented phase 5: see docker-compose.yml + §9a (networks agent, dockerctl)
    build: ./sandbox-manager
    environment: [DOCKER_HOST=tcp://socket-proxy-rw:2375]
    networks: [agent]
  socket-proxy-ro:
    image: tecnativa/docker-socket-proxy
    environment: [CONTAINERS=1, EVENTS=1, POST=0]
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]
    networks: [agent]
  socket-proxy-rw:
    image: tecnativa/docker-socket-proxy
    environment: [CONTAINERS=1, IMAGES=1, POST=1]
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]
    networks: [agent]
  syncthing:                             # vault sync to laptop/phone
    image: syncthing/syncthing
    volumes: ["/home/hoder/vault:/var/syncthing/vault", "syncthing:/var/syncthing/config"]
    ports: ["22000:22000/tcp", "22000:22000/udp", "127.0.0.1:8384:8384"]
    networks: [egress]
  # agent-worker additionally mounts:
  #   /home/hoder/vault:/vault:ro                 (indexer)
  #   /home/hoder/vault/Inbox:/vault-rw/Inbox     (writer)
  #   /home/hoder/vault/AI:/vault-rw/AI           (writer)
  # agent-api, agent-worker, media-worker mount /srv/agent/files:/files
  # agent-api, agent-worker, media-worker, sandboxes set:
  #   HTTPS_PROXY=http://egress-proxy:4750  NO_PROXY=litellm,ollama,agent-postgres,agent-redis,phoenix
  #   OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006
  # => remove `egress` from every service except egress-proxy and syncthing.
volumes: { ollama: {}, pg: {}, redis: {}, hf-cache: {}, syncthing: {}, phoenix: {} }
networks:
  agent:  { internal: true }
  egress: {}                              # outbound internet; only egress-proxy + syncthing attach
  edge:   { external: true, name: nodereact_edge }   # VERIFY: network shared with nodeReact nginx
```

nginx (nodeReact) additions: `location /api/agent/` and `location /line/webhook` → `agent-api:8000`, with `proxy_buffering off; proxy_read_timeout 3600s;` for SSE.

## 14. Repository layout

```
agent-harness/
  docker-compose.yml  .env.example  ARCHITECTURE.md
  config/  policy.yaml  models.yaml  privacy_rules.yaml  authority.yaml
  api/                 # FastAPI + LangGraph + arq tasks
    app/  main.py  routes/{web,line,events,ingest,uploads,todos,inbox}.py
          graphs/{main,grounded,grounded_state,chat,assistant,coding,verify}.py
          gates/{gate0,gate1_qwen,jev_router}.py  decision.py
          jobs/  memory/  rag/  tools/  mcp_server/  llm/
          tools/shop.py      # §12c store adapters (PChome, momo, Yahoo)
          tools/sandbox.py   # §9a sandbox-manager client; graphs/coding.py = coding route
    tests/
    langgraph.json
  media/               # media-worker
  governor/            # gpu-governor
  browser/             # §12b read-only renderer (Playwright/Chromium)
  sandbox-manager/     # app.py (FastAPI) + sandbox/Dockerfile (Copilot CLI image); repos in config/coding.yaml
  host/                # copilot hooks + install script
```

## 15. Build phases & acceptance criteria

**MVP = phases 1, 2, 3 + 8 (LINE).** Use the MVP daily for **2 weeks** before starting other phases;
re-prioritise remaining phases from real usage and the decision log.

| # | Phase | Done when |
|---|---|---|
| 1 | Compose skeleton: postgres, redis, ollama, **litellm**, **phoenix**, **egress-proxy**, agent-api; one ReAct chat graph; SSE; minimal React chat page in nodeReact | Chat streams tokens; thread survives agent-api restart; trace visible in Phoenix; non-allowlisted domain blocked |
| 2 | Gate 0 + Jev router + decision layer (+ Qwen fallback router; Gate 1 stub only); `decisions` logging + replay tool | Unit tests for `decide()` cover every rule; Jev outage → fallback works |
| 3 | Job layer (arq queues, cancel, dedupe) — chat already goes through it; **kill switch**; **ZFS backups** | Cancel button stops a run within one step; duplicate webhook ignored; `/stop cloud` blocks cloud calls; nightly backup + snapshot created |
| 4 | gpu-governor + Ollama unload — **implemented 2026-10-28** (§11; live GAMING test with `steam-headless` pending) | Starting `steam-headless` + streaming frees Qwen VRAM within 10 s; jobs route to cloud |
| 5 | Coding route: sandbox-manager + Copilot (headless, not ACP) + verify + approval — **implemented 2026-10-28** (§9a; verified end-to-end, squid gate passed) | Task produces diff, harness-run tests pass, push blocked until approved (nothing is ever pushed; approval creates a local branch) |
| 6 | Copilot ingest hooks + harness-memory MCP — **implemented 2026-10-30** (§9b; not yet run against a live SSH session) | SSH session appears as `copilot:<id>` thread; recall works from SSH Copilot |
| 7 | Todos + Inbox + background approvals ✅ | Todo CRUD from web, agent, Copilot; paused job resumes after approval days later |
| 8 | LINE channel + **health alerts** | Owner-only; quick reply + push for long jobs; no PRIVATE content sent; stopping a container triggers one LINE alert |
| 8a | Grounded answering (§12a): clarify interrupt/resume (web + LINE), claim extraction, authority tiers, per-turn caps — **implemented 2026-10-28** (web verified live; LINE resume not yet tested live) | §12a test table passes; ambiguous question asks before answering; every factual claim has one T1 citation or an "Unverified" label; caps never exceeded |
| 8b-read | Browser rendering fallback (§12b stage 1) — **implemented 2026-10-28** | JS-only page returns text via `browser`; private/loopback/metadata URLs blocked by squid; normal pages not rendered |
| 8b-login | Interactive browsing + owner login (§12b stage 2) — planned | §12b stage 2 acceptance list |
| 8c | Shopping: clarify item/budget, recommend real store listings (§12c) — **implemented 2026-10-28** (web verified live; LINE not yet tested live) | A missing budget is asked with option buttons; every pick is a real listing with code-rendered price/store/link; 「cheaper」 lowers the ceiling; unrelated listings are filtered |
| 9 | Media: voice, image, doc, then video | Each modality produces text; video runs as background job; CPU path works while GAMING |
| 10 | RAG + long-term memory + eval set + **memory review UI** — **implemented 2026-10-29** (§10; recall@5 = 1.00 reranked / 0.88 RRF-only on 33 questions; nodeReact `/Memory` page) | recall@5 measured on eval set; memories editable/deletable in nodeReact |
| 10a | Obsidian vault: Syncthing, indexer, writer tools, Notion import | Notion pages imported with no broken links; edits on phone re-indexed within 1 min; agent writes only to `Inbox/` and `AI/` |
| 10b | General L9 verify layer (§12): deterministic checks + model check (Jev/Qwen) + tier escalation (max 2) → Inbox item — planned. Today only coding verify (phase 5) and grounded claim verification (phase 8a) exist; the Inbox (phase 7) is ready to receive the failure items | A failed check escalates local → cloud_fast → cloud_strong at most twice, then creates an Inbox item; private runs never use cloud; extra model cost measured on the eval set |
| 11 | Cron/events, eval suite in CI, monthly restore test | Morning brief runs; eval regression check; restore test passes |

## 16. Security checklist

- No agent container mounts `csr/`, `.env*`, home dir, or the raw Docker socket.
- Only `sandbox-manager` can create containers (via `socket-proxy-rw`).
  - **Risk:** `socket-proxy-rw` allows `POST` on containers, which is effectively root on the host (anyone who can
    reach it can start a privileged container). It therefore sits alone on the internal `dockerctl` network with
    sandbox-manager; sandbox-manager accepts only the `HARNESS_TOKEN` bearer and builds every container spec itself
    (fixed image, uid 1000, `cap_drop: ALL`, no privileged mode, mounts only inside its task dir).
  - Coding sandboxes see only their clone; `COPILOT_GITHUB_TOKEN` (Copilot Requests only) goes to Copilot containers
    and never to test containers. No push credentials exist; approval only creates a local branch.
- Secrets in `.env` (gitignored) or Docker secrets; injected into tool code; never placed in prompts.
- All content from mail, web, documents, vault/Notion, LINE, and media is **untrusted data** (prompt-injection defence):
  - Every tool result and retrieved chunk carries `trust: owner|untrusted`; only direct owner messages are `owner`.
  - Untrusted content is wrapped in delimiters (`<untrusted source="…">…</untrusted>`) with a system rule that it is data, never instructions.
  - **Taint tracking:** once any untrusted content enters a run, the run is `tainted`. In a tainted run, every tool call
    classed `risky` or `external` (send, write outside `Inbox/`/`AI/`, push, delete, network, money) requires `interrupt()` approval,
    regardless of the router's `risky` score.
  - Tool allowlist per route; MCP servers allowlisted; tool arguments validated (paths confined, URLs via egress proxy).
  - Data from PRIVATE sources may not be passed to tools that send to the outside world in the same run without approval.
- LINE: signature verification + owner allowlist. Events: HMAC. Web: nodeReact auth.
- `agent-api` host port bound to `127.0.0.1` only.
- Browsers (§12b): no host ports, internal network only, egress only via squid (private IPs denied); the read-only renderer keeps no state between requests and never holds logins.
- Audit log of every tool call and decision in Postgres.

## 17. Open items to VERIFY

- `typesafe-sdk` async client and response field names; Jev data retention terms.
- Qwen3.5 9B Ollama tag and whether it supports vision.
- LINE reply-token expiry and content retention window.
- Copilot CLI `--allow-tool` / `--deny-tool` pattern syntax (resolved 2026-10-28: `copilot help permissions` —
  `shell(cmd)`, `shell(git:*)` prefix form, `write(path?)`, `url(…)`; deny beats allow); ACP permission message format; terms for non-coding use.
- LiteLLM, Phoenix, Smokescreen image tags and config keys.
- Copilot CLI: honours `HTTPS_PROXY` (resolved 2026-10-28: all Copilot traffic seen in the squid log);
  least-privilege token type; usage/credit field names (resolved: `session.usage_checkpoint`).
- Name of the nodeReact nginx Docker network.
- Obsidian Importer: Notion API support and database → Bases mapping; linuxserver/obsidian image.
- §12a: Brave `freshness` parameter and per-query price; whether the LINE reply token outlives a clarify round (else use push); local KB (§10a) timeline for PRIVATE verification.
  - Resolved 2026-10-28 (langgraph 1.2.16):
    - interrupts arrive as `updates` `{"__interrupt__": (Interrupt(...),)}`;
    - `Command(resume=None)` re-raises the interrupt, so the sentinel `"/skip"` is used;
    - new input on a thread with a pending interrupt restarts from START, so no `aupdate_state` is needed.
- §11: whether `capabilities: [utility]` alone gives NVML on every driver. It works on the RTX 5060 Ti host as of 2026-10-28; fall back to `[gpu, utility]` if not.

## 18. Adopted improvements (decided 2026-10-20)

| # | Improvement | Decision | Where |
|---|---|---|---|
| 1 | LLM gateway | **Adopted — LiteLLM** | §2, §7, §13 |
| 2 | Observability | **Adopted — Arize Phoenix** | §2, §13 |
| 3 | Replayable decision log | **Adopted** | §7, §19 |
| 4 | Jev shadow test vs Qwen router | **Rejected** — Jev is the router; Qwen router is fallback only | §6 |
| 5 | Start without Gate 1 | **Adopted** | §6 |
| 6 | File storage | **Plain volume first** (no MinIO) | §2, §5, §13 |
| 7 | Backups | **ZFS HDD pool `tank`** | §19 |
| 8 | Kill switch | **Adopted** | §19 |
| 9 | Memory review UI | **Adopted** | §19 |
| 10 | Health checks + LINE alerts | **Adopted** | §19 |
| 11 | Egress allowlist proxy | **Adopted** | §2, §13 |
| 12 | Grounded answering: clarify first, one primary source per claim, labelled partial answers (2026-10-22) | **Adopted** | §12a |

## 19. Operations

### Decision replay
- `decisions` table stores: job_id, redacted input text, gate0 result, raw Jev output, gpu_state, channel, mode, budget snapshot, policy version, decision, outcome (verify pass/fail, escalations, user correction), latency, cost.
- `python -m app.tools.replay --policy config/policy.next.yaml --since 14d` re-runs `decide()` on stored inputs (no model calls) and prints a diff: route/model changes, estimated cost change, and cases where the old decision failed verification.
- Policy files are versioned; a new policy is deployed only after a replay review.

### Backups (ZFS)
- Pool `tank` is a **single HDD (`sdb`) with no redundancy** → protects against mistakes/corruption, **not** disk failure. Recommended: add a second disk as a mirror (`zpool attach tank sdb <disk>`; free `sda` 5.5 TB may qualify — VERIFY it holds no data) and/or an offsite copy later.
- Create dataset: `zfs create -o compression=zstd -o mountpoint=/tank/agent-backup tank/agent-backup` (host, root, one-off).
- Nightly host cron / systemd timer (`host/backup.sh`):
  1. `docker exec agent-postgres pg_dump -Fc` → `/tank/agent-backup/pg/agent-YYYYMMDD.dump`
  2. `rsync -a --delete` of `/srv/agent/files`, `/home/hoder/vault` (+ `.git`), `~/agent-harness/config`, Redis AOF, Phoenix data → `/tank/agent-backup/…`
  3. `zfs snapshot tank/agent-backup@YYYYMMDD`
  4. Retention: 14 daily, 8 weekly, 6 monthly snapshots (sanoid or simple script).
- Never back up `.env` in plain text; store it encrypted (e.g. `age`) if included.
- Monthly restore test: restore latest dump into a throwaway `pgvector/pgvector:pg17` container and run a smoke query; result → LINE alert.

### Kill switch
- Redis key `killswitch` ∈ {off, cloud, all}. `cloud` = block every non-local model call (LiteLLM pre-call hook + decision layer) and Copilot sandboxes; `all` = also pause every background queue.
- Controls: nodeReact UI toggle, LINE command `/stop cloud`, `/stop all`, `/resume` (owner only), and `host/killswitch.sh`.
- Auto-trigger `cloud` when daily spend hits 100% of budget.

### Copilot credit tracking
- Copilot calls bypass LiteLLM, so track them separately:
  - Sandbox runs: always pass `--max-ai-credits`; record usage from Copilot OTel metrics (exported to Phoenix) and/or
    the JSON output stream at run end → `usage(provider='copilot', credits, run_id)` table.
  - SSH sessions: ingest worker reads usage from session events (`assistant_usage_events` / usage events; VERIFY field names).
- Budget check in Gate 0 and the kill switch use **LiteLLM spend + Copilot credits** (separate daily limits).
- At 80%: LINE alert; at 100% of the Copilot limit: coding route for harness-launched runs is blocked (SSH use is unaffected but alerted).

### Memory review UI (nodeReact)
- Page listing `PostgresStore` items by namespace with search, edit, delete, and "pin".
- Episodes expire after a configurable TTL (default 180 days) unless pinned.
- Conflicting facts: newest wins, previous kept as history; UI shows history.
- Every agent memory write is visible in the Inbox as a low-priority item. (Implemented 2026-10-29 as nodeReact `/Memory`: memories plus a knowledge-base tab; Inbox items: phase 7.)

### Health checks & LINE alerts
- Compose `healthcheck` on every service; `alert-watcher` job (agent-scheduler, every minute) checks: unhealthy containers, queue age > threshold, jobs stuck > timeout, LiteLLM spend > 80%, backup failure, disk/ZFS pool > 85% or not ONLINE, gpu-governor heartbeat missing.
- Alerts → LINE push to owner, deduplicated (one alert per issue per hour) + logged to Inbox.

### Egress allowlist (`config/egress-allowlist.yaml`)
- Allowed: TypeSafe API, OpenAI, Anthropic, LINE API/content, GitHub (api, github.com, Copilot endpoints — VERIFY list), package registries (npm, PyPI) for sandboxes, Hugging Face (model downloads, can be disabled after setup), Ollama registry.
- Everything else denied and logged; denials appear in Phoenix/alerts.


## Implemented 2026-10-30 (phase 7: Todos + Inbox)

- **Todos:** table `todos` (`app/todos.py`). REST `/api/agent/todos` (GET `?status=open|done|all`, POST, PUT, DELETE); agent tools `list_todos`/`add_todo`/`complete_todo` (assistant route only, no delete); MCP `list_todos`/`add_todo`. Due dates are Asia/Taipei; a bare date means 23:59. Due reminders: see below.
- **Inbox:** table `inbox` (`app/inbox.py`), kinds `approval|question|alert|memory`. REST `/api/agent/inbox` (list+counts, `/count`, `/{id}/read`, `/{id}/dismiss`, bulk `/dismiss` for alert|memory only, `/{id}/answer`).
- **Approvals days later:** coding approvals and background questions create an item (LINE push unless it came from LINE; private threads get a generic text). The LangGraph checkpoint lives in Postgres; if the Redis pending marker expired, `answer` restores it and resumes. It refuses (409) if the thread has moved on. Dismissing an approval = reject (sandbox discarded).
- Alerts and memory writes feed the Inbox; `inbox_cleanup_cron` (04:00) deletes old done/low items (`inbox:` in policy.yaml).
- nodeReact pages `/Todos` and `/Inbox` (approve/reject/answer, dismiss, bulk-dismiss memory/alert). Not done: verify-escalation → Inbox (needs the general L9 layer, phase 10b).

### nodeReact tooling notes (found during phase 7)

- **Run everything on Node 14:** build, tests and commands for nodeReact go in a Node 14 container (`node:14-bullseye-slim`, git installed, `npm ci --legacy-peer-deps`), never on the host's Node 22 (see `nodeReact/AGENTS.md`).
- **Front-end tests pass on Node 14 as-is:** `npx jest --config jest.front.cjs --forceExit src/front` gives 18 suites / 1331 tests with the original lockfile. The `@babel/core@7.12.10` failure only happened on host Node 22, so no babel upgrade is needed (an upgrade would rewrite the lockfile to v3 and break plain `npm ci`).
- **Release build works on Node 14** (`npm run build`, emits `release.js`, checked with the phase 7 pages). The host Node 22 failures (chunk conflict, OpenSSL) are a host-toolchain problem.
- **Routing fix (found live):** with `grounding.enabled` every non-coding run used the grounded graph, which has no todo tools, so the agent could not manage todos. The assistant route now uses the ReAct graph (`tools_for`), verified live: "add a todo..." created todo #4 with `source_channel=agent`.

### Todo due reminders (phase 7 addition)

- Cron `todo_reminders_cron` (every minute, `jobs/todo_jobs.py`); `todos.reminders_sent` (0 none, 1 due soon, 2 overdue) makes each stage fire once (atomic claim). Changing `due_at` or reopening a todo resets it.
- Stage 1 `before_s` (1 h) before `due_at`, stage 2 `overdue_s` (1 h) after it if still open; a todo created long overdue gets only the overdue reminder. Config: `todos.reminders` in policy.yaml.
- Quiet hours: Mon-Fri 00:00-07:00 (Asia/Taipei); nothing is sent or claimed then, so reminders go out right after 07:00. Weekends have no quiet window.
- Privacy: LINE gets only a generic line ("N todo reminders (K overdue). Open the Inbox..."), never a todo title; the full title is an `alert` item in the Inbox (owner-only web). Per-todo private flag not built.
- Not done: recurring todos. Verified live with a todo due in 30 min: one Inbox alert, no duplicate on later ticks.

- **Live check of "approval resumes after days" (2026-10-30):** a real `/code` run paused at the approval; I deleted its Redis `pending:` marker (what expiry does), then dismissed the Inbox item. The marker was restored, the checkpoint resumed from Postgres, the run replied "Discarded; nothing was committed", the item became `done`, and no branch or sandbox clone was left. The Approve path (`/inbox/{id}/answer`) shares `submit_answer` but was only unit-tested, not run live.

- **LINE todo fixes (2026-10-30, found when the owner added a todo from LINE):** (1) local Qwen sometimes ends a tool run with an empty message, which LINE showed as "(empty)"; the worker now replies with the last tool result of the turn (`_tool_fallback`). (2) The same model computed "tomorrow" wrongly (due one day late), so `parse_due` resolves `today`/`tomorrow`/`明天`/`後天` (+ optional HH:MM) itself and the tool says not to compute dates. (3) `add_todo` echoes the due date (or "no due date") and ignores an identical call within 10 minutes (the model sometimes repeats a call). Not fixed: a local model can still claim success without calling the tool (seen once in 5 runs); the tool result line is the ground truth.

## Implemented: phase 7b — Google Calendar (2026-10-30)

- `app/gcal.py` (client, refresh-token flow, in-memory access token) and `app/tools/calendar.py`. Grant file written by `host/google_auth.py` (PKCE, paste-the-redirect, no port-forward), mounted ro at `/google`. Egress allowlist gained `www.googleapis.com` and `oauth2.googleapis.com`.
- Privacy: `private_keywords` (行事曆/日曆/行程/calendar) force PRIVATE; `tools_for(..., local=True)` is the second guard, so cloud models never get calendar tools. Event text is wrapped in `<untrusted source="calendar">`.
- Approval design: instead of a graph `interrupt()`, add/delete create an Inbox `approval` item (`detail.calendar`, no run) and the Inbox `answer` route executes it. The existing Inbox UI needed no change; Reject = dismiss. Approving twice is refused (item no longer open).
- Verified live: read of the real calendar, `/local` run proposed an event (Inbox item), Approve created it in Google, then deleted. The delete-proposal path is only covered by tests. The OAuth app is unverified-production (100-user cap, no 7-day token expiry).

- **Calendar vs todo on LINE (2026-10-30):** local Qwen filed every "add ..." as a todo, worse on long LINE threads (it copies earlier tool calls). Fixes, in order of effect: `tools.narrow_for_intent` (deterministic, no model: add verb + time of day and no todo word → `add_todo` removed; explicit todo/reminder → `add_event` removed); the assistant route hides earlier turns' tool calls and keeps `executor.assistant_history_turns` (8) turns (`graphs.main.drop_old_tool_turns`); the prompt rule. A Jev/cloud classifier was rejected: the text is PRIVATE. LINE gets a content-free notice when a proposal was made.
- **Claimed-success guard (2026-10-30):** on LINE, Qwen answered "proposal created" after only `get_current_time`. `tools.required_tool` names the write tool a clear request must call (add verb + time of day → `add_event`; 新增待辦 → `add_todo`); `build_graph(require_tool=...)` nudges the model once if it tries to finish without calling it, then replies that the action failed. Private LINE notices (`_private_line_notice`) report what the turn really did.

## Tool-call reliability with the small local model (2026-10-30)

Local Qwen runs the assistant route and PRIVATE content, so it cannot be replaced by a cloud classifier. Its failures
were all seen live (web and LINE); the fixes are deterministic code around the model, not better prompting alone.

| Failure seen | Layer that handles it | Where |
|---|---|---|
| Picks the wrong tool (every "add ..." became `add_todo`, copied from earlier turns) | **Narrow the tool list before the run**: add verb + time of day and no todo word → `add_todo` removed; todo/reminder words → `add_event` removed; ambiguous → both kept | `tools.narrow_for_intent` |
| Copies tool calls from long history | Assistant route keeps `executor.assistant_history_turns` turns and hides earlier turns' tool calls/results from the model input (checkpoint untouched) | `graphs.main.drop_old_tool_turns` |
| Says "done" without calling any tool (calendar proposal, todo) | **Required-tool check**: for a clear request `required_tool` names the write tool; if the model tries to finish without it having been called this turn, nudge once ("nothing was saved, call X now"); if still no call, the reply says the action failed | `tools.required_tool`, `build_graph(require_tool=...)` |
| Wrong or invented arguments (relative dates, "tomorrow+1 day", year 2023) | Tools take words and resolve them in code: `today/tomorrow/day after tomorrow/今天/明天/後天` + HH:MM, `MM/DD`, `MM月DD日`; bad values return "Not added: ..." rather than guessing | `todos.parse_due`, `gcal.when` |
| Repeats the same call | Duplicate guards: identical todo within 10 min; identical open calendar proposal | `tools/todos.py`, `tools/calendar.py` |
| Empty final message after a tool result | Reply with the last tool result of the turn | `worker._tool_fallback` |
| Owner cannot see PRIVATE answers on LINE, so cannot tell what happened | The LINE notice is built from this turn's real tool results (added / proposed / failed / nothing happened / the assistant is asking you a question that must be answered on the web). No content and no URLs: it only names the page (Todos, Inbox) | `worker._private_line_notice` |

**Rule for new tools:** a tool that writes anything (todo, calendar, memory, later email) must (1) resolve user-facing
values in code, (2) return a one-line result that starts with a stable prefix (`Added`, `Proposed`, `Not added`) so
those prefixes can drive the checks above, (3) be covered by `narrow_for_intent`/`required_tool` when the request is
recognisable by cheap rules, and (4) never be reported as done unless its tool result says so. Intent rules are
regexes in `tools/__init__.py`; keep them conservative (queries such as "what todos do I have" are excluded) and leave
both tools available when the request is ambiguous.

**Not done / limits:** the rules only cover clear phrasings; an ambiguous message ("新增 買牛奶") can still be filed in
the wrong place. A general model-based verifier for tool use belongs to phase 10b (L9 verify layer), and must stay
local for PRIVATE text. Qwen still occasionally writes Simplified Chinese despite the prompt rule.

- **LINE is chat + notify only (2026-10-30):** `decision` rule 6 forces every LINE message to the `chat` route (no todo/calendar/coding tools; models stay local or cloud_fast). A clear add-todo/add-event phrasing on LINE is answered in code ("use the web") before any model runs (`pipeline._line_write_request`), so the model cannot claim it saved something. Todos, calendar and approvals are done on the web. `GET /threads` now returns `private`; the web chat list shows 🔒 and the header shows "Private · local only" for a locked thread (a new chat unlocks).
