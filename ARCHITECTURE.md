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
}
```
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
    # 3. intent confidence < 0.5 → route="clarify"
    #    0.5 ≤ conf < 0.8 → bump tier one level
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

## 12. L9 Verify layer & L10 Approval/Output

Verify (skip for trivial chat):
1. Deterministic: coding → harness-run tests/lint/build; assistant → schema valid, no tool errors, citations present when RAG used.
2. Model check: Jev `Noul("The response fully and correctly answers the request")` for non-private; Qwen for private; deterministic-only when GAMING + private.
3. Fail → escalate tier (local → cloud_fast → cloud_strong), max 2 escalations → Inbox item.

Approval:
- `interrupt()` before: git push/PR merge, send mail/message to others, delete, calendar create/delete, any money action, network enablement in sandbox, any action flagged `risky`.
- Background approvals persist (Postgres checkpoint) → Inbox card in nodeReact + LINE push notification → resume on approval.

Output: SSE (web), LINE reply/push (never PRIVATE content), Inbox, memory write, PR link.

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
  sandbox-manager:
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
  config/  policy.yaml  models.yaml  privacy_rules.yaml
  api/                 # FastAPI + LangGraph + arq tasks
    app/  main.py  routes/{web,line,events,ingest,uploads,todos,inbox}.py
          graphs/{main,chat,assistant,coding,verify}.py
          gates/{gate0,gate1_qwen,jev_router}.py  decision.py
          jobs/  memory/  rag/  tools/  mcp_server/  llm/
    tests/
    langgraph.json
  media/               # media-worker
  governor/            # gpu-governor
  sandbox-manager/     # + sandbox image Dockerfile
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
| 4 | gpu-governor + Ollama unload | Starting `steam-headless` + streaming frees Qwen VRAM within 10 s; jobs route to cloud |
| 5 | Coding route: sandbox-manager + Copilot ACP + verify + approval | Task produces diff, harness-run tests pass, push blocked until approved |
| 6 | Copilot ingest hooks + harness-memory MCP | SSH session appears as `copilot:<id>` thread; recall works from SSH Copilot |
| 7 | Todos + Inbox + background approvals | Todo CRUD from web, agent, Copilot; paused job resumes after approval days later |
| 8 | LINE channel + **health alerts** | Owner-only; quick reply + push for long jobs; no PRIVATE content sent; stopping a container triggers one LINE alert |
| 9 | Media: voice, image, doc, then video | Each modality produces text; video runs as background job; CPU path works while GAMING |
| 10 | RAG + long-term memory + eval set + **memory review UI** | recall@5 measured on eval set; memories editable/deletable in nodeReact |
| 10a | Obsidian vault: Syncthing, indexer, writer tools, Notion import | Notion pages imported with no broken links; edits on phone re-indexed within 1 min; agent writes only to `Inbox/` and `AI/` |
| 11 | Cron/events, eval suite in CI, monthly restore test | Morning brief runs; eval regression check; restore test passes |

## 16. Security checklist

- No agent container mounts `csr/`, `.env*`, home dir, or the raw Docker socket.
- Only `sandbox-manager` can create containers (via `socket-proxy-rw`).
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
- Audit log of every tool call and decision in Postgres.

## 17. Open items to VERIFY

- `typesafe-sdk` async client and response field names; Jev data retention terms.
- Qwen3.5 9B Ollama tag and whether it supports vision.
- LINE reply-token expiry and content retention window.
- Copilot CLI `--allow-tool` / `--deny-tool` pattern syntax; ACP permission message format; terms for non-coding use.
- LiteLLM, Phoenix, Smokescreen image tags and config keys.
- Copilot CLI: honours `HTTPS_PROXY`; least-privilege token type; usage/credit field names in events and OTel.
- Name of the nodeReact nginx Docker network.
- Obsidian Importer: Notion API support and database → Bases mapping; linuxserver/obsidian image.

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
- Every agent memory write is visible in the Inbox as a low-priority item.

### Health checks & LINE alerts
- Compose `healthcheck` on every service; `alert-watcher` job (agent-scheduler, every minute) checks: unhealthy containers, queue age > threshold, jobs stuck > timeout, LiteLLM spend > 80%, backup failure, disk/ZFS pool > 85% or not ONLINE, gpu-governor heartbeat missing.
- Alerts → LINE push to owner, deduplicated (one alert per issue per hour) + logged to Inbox.

### Egress allowlist (`config/egress-allowlist.yaml`)
- Allowed: TypeSafe API, OpenAI, Anthropic, LINE API/content, GitHub (api, github.com, Copilot endpoints — VERIFY list), package registries (npm, PyPI) for sandboxes, Hugging Face (model downloads, can be disabled after setup), Ollama registry.
- Everything else denied and logged; denials appear in Phoenix/alerts.
