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
| ASR | faster-whisper large-v3-turbo (chosen, phase 9); OpenCC → Traditional Chinese |
| OCR | RapidOCR 3.x, PP-OCRv6 small (one model: Simplified, Traditional, Japanese, English; onnxruntime CPU, baked into the image) |
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
| Cron | arq cron `schedules_cron` reads the `schedules` table every minute (Asia/Taipei) | internal | morning brief, nightly eval, owner prompts (§19) |
| Events | `POST /events/{source}` | per-source HMAC (`X-Hub-Signature-256`) | GitHub webhook etc. → Inbox item only, never an agent run (§19) |
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
| image | OCR + local VLM caption (Qwen3.5 9B vision via Ollama, `think:false` — resolved 2026-11-04) | OCR on CPU; vision → cloud or defer | GPT/Claude vision |
| video | ffmpeg → audio→ASR; scene keyframes (max 16–32) → VLM; build timeline text | CPU, background only | frames + transcript to GPT/Claude |
| doc | Docling → markdown; >~20 pages → chunk + index into RAG | Docling on CPU | Claude/GPT for hard layouts |

4. Strip EXIF/GPS before any cloud upload.
5. Output: `{text, attachments[], modality, privacy_tag, summary}`.
6. Video and large docs are always `mode=background`; the user gets an immediate "processing" reply.

### Implemented 2026-11-04 (phase 9)

- **Flow.** Web: `POST /api/agent/uploads` (raw body, `X-Filename`, `?thread=`) stores the file, inserts the
  `attachments` row and starts extraction at once (`extract_attachment`, job `extract:{id}`, queue `arq:media`); the
  run then sends `attachments: [ids]` (text may be empty). LINE: image/audio/video/file messages are downloaded from
  the content API (dedupe first, so redelivery never re-downloads). Both call `media.start_run`: run meta
  `processing`, job `media:{run_id}` on the media-worker, which extracts every file (sha256 cache: a done,
  non-degraded row with the same hash is reused; a Redis lock stops two workers doing the same file) and enqueues
  `submit_media_run` (`msub:{run_id}`, interactive or background queue) on the agent worker. That composes the run
  text and calls `pipeline.submit(run_id=…, router_text=…)`, so one run_id spans upload → answer. SSE adds
  `status {processing, step}` and `attachment {id, kind, status, summary}`; cancel works while processing.
- **Run text.** The owner's words (or `media.default_prompts[kind]` when empty) followed by one
  `<untrusted source="attachment" kind=… name=…>` block per file (`</untrusted` inside the text is escaped), capped by
  `media.max_chars`. A short voice note (≤ `voice_as_message_s`, or any LINE voice) with no typed text *is* the
  request. Thread history shows `📎 name` instead of the blocks (`media.display_text`).
- **Routing.** The router, decision log and thread context see only the instruction plus `📎 kind: name`
  (`media.router_view`), never file content: a long video timeline drowned the instruction (intent 0.31 → clarify,
  risky 0.65) and file text must not steer routing. Gate 0 still runs on the full text, so PRIVATE content in a file
  locks the run local. Decision rule 3 never clarifies a run with attachments (the file is the subject). Rule 9 sends
  video and `large_doc` to the background queue.
- **Injection guard (§16).** File text never triggers actions. Live, "summarise this docx" created a todo from the
  meeting minutes; now `tools.for_attachments` drops write tools unless the owner's own words ask for that action,
  `narrow_for_intent`/`required_tool` look only at the owner's words, the pipeline shortcuts (`/commands`, `記住`,
  the LINE write refusal) use `media.strip_blocks(text)`, and memory extraction gets the stripped text.
- **media-worker** (`media/`, own image ~7.8 GB: torch CPU, ctranslate2 4.8 + nvidia cuBLAS/cuDNN pip libs, ffmpeg,
  Docling, RapidOCR, OpenCC; models in volume `media-models`; files mounted read-only):

  | Modality | GPU FREE | BUSY / GAMING | Notes |
  |---|---|---|---|
  | voice | faster-whisper large-v3-turbo CUDA fp16 under the GPU semaphore (7.6 s clip in ~1 s) | CPU int8 (4.2 s) | ffmpeg → 16 kHz WAV → numpy (this PyAV build rejects faster-whisper's `metadata_errors`); s2twp; GPU model unloaded after `asr.keep_alive_s` or when the GPU leaves FREE |
  | image | RapidOCR + Qwen caption | OCR only, `degraded: true` (re-extracted next time) | EXIF-transposed, re-encoded without EXIF before any model; `image.when_gpu_busy: ocr_only` default; `cloud` = NORMAL files only, cost recorded from LiteLLM |
  | doc | pdfium text; < 30 chars/page → render + OCR (scans); Docling for docx/xlsx/pptx and PDFs ≤ 50 pages | same (CPU) | > 20 pages or > 40 000 chars → `large_doc`: indexed into the KB (`upload/…md`) so answers cite it via RAG |
  | video | ffmpeg audio → ASR; scene cuts on a 320-px proxy, up to `video.max_frames` (16) → caption + OCR; timeline `[mm:ss] 🗣 …` / `🖼 …` | CPU, background | audio-only containers become `voice` |

  `killswitch=all` makes heavy extraction retry later; `media.max_jobs` (2) bounds concurrency.
- **Verified live 2026-11-04:** receipt image (GPU caption + OCR, NT$245 answered); voice → todo request (GPU 6.3 s
  incl. load; CPU int8 4.2 s while GAMING); docx summary with no todo created; scanned PDF (OCR path); 25-page text
  PDF → `large_doc`, indexed, "parking cap" answered from page 17 with a KB citation; 14 s video → background queue,
  5 frames + transcript, correct summary; GAMING: image OCR-only in 0.4 s, routed `cloud_fast` by rule 5. LINE:
  content API reachable through squid (real 404 for a bad id; download errors now reply "無法下載…"); a full LINE
  media round trip is covered by tests only.
- **LINE photo + question (2026-11-04, found live):** LINE delivers a photo and the text typed after it as two events,
  so "這裡面是什麼" ran without the photo and was clarified, while the photo got the default prompt. Now a LINE
  file sent without text (not voice) is noted for `media.line_follow_s` (60 s) and its hand-off waits until
  `line_text_wait_s` (15 s) after it arrived (`_defer_by`). The next text (not a `/command`, not an answer to a
  pending question) is offered to that run (`media:text:{run_id}`, SET NX); `submit_media_run` closes the slot and
  uses it as the instruction with the text's fresh reply token. If the run was already submitted, the text starts a
  new run with the same file attached (extraction cached). Only one text per file; multi-photo albums are separate runs.
- **Shelf photo, "which is cheapest per gram" (2026-11-04, found live):** the answer model only had flat OCR rows, so
  prices floated free of products; it was asked "factual or shopping?", escalated to `cloud_strong` (verify 0.11) and
  hedged at $0.037. Fixed:
  - *Pictures to cloud tiers.* The worker puts the turn's image ids in grounded state (`images`). For `cloud_*`
    models on NORMAL runs, `draft` and `compose` attach them (`media.image_parts`: NORMAL images only,
    re-encoded at `image.cloud_max_side` 2048 px without EXIF, at most `image.max_images`, built at call time so
    no base64 reaches checkpoints). Local tiers and PRIVATE runs get text only; `image.send_to_cloud: false` turns it off.
  - *Price pairing.* OCR keeps the boxes. Each line stating a size (25G, 500ml…) is paired one-to-one with the
    nearest bare price box that is printed bigger than the product line (shelf tags; keeps package numbers out).
    Look-alike digits are mapped and flagged ("IZO" → 120) as a fallback. The pairs go in a 【標價配對】 section,
    so local-only runs can read them too.
  - OCR switched from RapidOCR 1.4 (PP-OCRv4 Chinese, mostly Simplified, misread kana and price digits) to
    RapidOCR 3.x PP-OCRv6 small, which prints Japanese and Traditional as seen. It is 0.5 s on the shelf photo
    and the models are baked in at build time. A line is converted with s2tw only if it has no kana and contains
    a Simplified-only character that Japanese never uses (not encodable in Shift-JIS: 绳, 经, 带), so 干物 and
    国内製造 stay as printed. The caption prompt says not to add currency, units or brands that aren't visible
    (it had written 日元).
  - Rule 3b: with attachments, a near-tie between task types is not asked. The type that answers from the
    files wins over a web pipeline (shopping 0.46 vs factual 0.35 → factual).
  - Re-run: correct (45 g / 98 → about 2.18 per gram), `cloud_fast`, verify 0.77, $0.0008, 14 s, no question asked.
- **Not done:** cloud transcription / cloud vision fallbacks for voice and video (local CPU is always available);
  HEIC is decoded but not tested live; PDFs > 50 pages skip Docling (plain per-page text).

## 6. L3–L5 Gates

### Gate 0 — deterministic (pure code, must be < 5 ms)
- Honour user override (`/local`, `/cloud`, `/code`).
- Owner commands are handled before Gate 0 (`pipeline.handle_command`), never reach a model, and reply in English:
  `/stop cloud|all`, `/resume`, `/status`, `/help`. Any other leading `/word` (e.g. `/new`) replies "Unknown command" + the
  `/help` list. Paths such as `/etc/hosts …` are not commands.
- Secret/PII regex scan (API keys, tokens, private keys, ID numbers, card numbers, phone, email). Redact for any non-local use.
- **Source tags:** content originating from mail, finance docs, `~/private`, private vault notes ⇒ that message runs on a
  local model only (`local_only`, since 2026-11-06; see "Privacy model revised" below). Only an owner-marked thread
  gets `privacy_tag=PRIVATE`.
- **LINE is never PRIVATE (2026-10-30):** for `channel="line"` Gate 0 always returns no privacy tag (sources, patterns
  and the thread mark are ignored); secrets and PII are still redacted before any cloud call. Only web
  threads can be PRIVATE. "Never send PRIVATE content to LINE" below therefore means content from PRIVATE *web*
  threads (notices, approvals, reminders), which LINE receives only as content-free notices.
- Budget check (daily $ per provider, Copilot AI credits).

### Gate 1 — Qwen privacy check (DEFERRED — not in initial build)
Ship without Gate 1. Add it only if the decision log shows Gate 0 missing private data.
Keep the interface (`gate1(text) -> is_private`) as a no-op stub returning `None`.
- Only when Gate 0 is inconclusive on free-form text. Output: `is_private: bool`.
- When `gpu_state=GAMING`: skip and treat inconclusive as PRIVATE for background jobs.

### Privacy model revised — secrets + owner choice only (decided 2026-11-05, implemented 2026-11-06)
Measured over 14 days: 20 of 367 requests were PRIVATE, 17 of them only because the thread had once contained
PRIVATE content, the rest from keywords (`行程` also matched travel itineraries). Local models are much weaker than
cloud, so PRIVATE should be rare and deliberate. New rules:
- **Secrets/PII are always redacted** before any cloud call (Gate 0 regex, unchanged). A `private: true` pattern
  (secrets such as keys, private keys, tokens) still forces local for that message only.
- **PRIVATE only by owner decision:** `/local`, a thread the owner marks private (toggle in nodeReact chat, stored
  in `thread_private:*`), a vault note with frontmatter `privacy: private`, files under `~/private` / `private/`
  doc dirs, and explicit `private_paths`. These stay local-only as today (`decision.py` rule 2).
- **Removed:** `private_keywords` (calendar terms), content-based thread stickiness (a thread no longer becomes
  PRIVATE because one message was; `pipeline.py` stops calling `mark_thread_private` on Gate 0 hits),
  `private_sources` for calendar/personal notes.
- **Opened to cloud:** calendar tools (`tools/__init__.py` offers them on cloud routes too; event approvals stay in
  the Inbox), `AI/Daily/` notes (no longer forced `privacy: private`; still private if the day includes a PRIVATE
  thread), episodes of non-private threads.
- Unchanged: LINE never receives PRIVATE content; morning brief LINE text stays content-free; existing PRIVATE
  threads/notes keep their tag (no automatic downgrade).
- Gate 1 is not needed for this model; it stays deferred.

Implementation (2026-11-06):
- `gate0()` returns `privacy_tag=PRIVATE` only for an owner-marked thread. A `private: true` pattern or a
  `private_sources` hit sets `Gate0Result.local_only`: `decision.py` rule 1b pins that message to `local` (beats
  `/cloud`; `/code` becomes `chat`, Copilot is cloud), but the run is **not** private. Its stored text is redacted and
  it gets no PRIVATE notes/memories, so a later cloud turn in the same thread cannot see anything private.
  `/remember` and `/note` still store a local-only fact/note as private; outgoing web queries with a hit are blocked.
- Only owner-marked PRIVATE threads keep unredacted text in their history (`pipeline.submit`).
- `PUT /api/agent/threads/{tid}/private {"private": true}` marks a thread (audited, kind `privacy`). One-way: `false`
  on a PRIVATE thread returns 409 (its history holds unredacted text); the LINE thread returns 400. nodeReact chat
  shows a 🔓 button (confirm dialog) on web chats, including a new chat before its first message.
- `private_keywords` is gone from `privacy_rules.yaml`, `gate0.py` and `rag/store.doc_privacy`; `private_sources`
  dropped `calendar` and `personal_notes`. Documents already indexed as PRIVATE by a keyword keep that tag until
  they change.
- Calendar tools are offered on every assistant run (`tools_for` lost its `local` argument); approval items are no
  longer private and the LINE notice no longer says "private".
- Morning brief: the `web:brief` thread is no longer marked private and its Inbox item is normal (owner decision
  2026-11-06); the existing mark was cleared once by hand.
- `AI/Daily/`: `vault_daily.run` writes `privacy: private` only if one of that day's episodes is private; a note
  already private stays private (`_flip_private` only upgrades).

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
    # 3. intent confidence < 0.7 (`thresholds.clarify_below`) OR router.ambiguous > 0.7 → route="clarify" on a LOCAL model (local_cpu if GPU BUSY/GAMING);
    #    no tier bump — never pay a cloud model to ask a question. /cloud override and PRIVATE keep their model.
    #    The answer is re-routed together with the original request + question (§12a MVP), so the tier
    #    (and any escalation) is decided once the intent is clear. After `clarify.max_rounds` (10) clarify rounds
    #    → answer with stated assumptions.
    # 4. code_multi > 0.6 or long_ctx > 0.6 → cloud_strong (cloud_fast if over budget)
    # 5. gpu_state in (BUSY, GAMING) and model == local and not PRIVATE → cloud_fast
    # 6. channel == "line" → route is always "chat" (assistant/coding/clarify become chat: LINE is chat + notify only,
    #    todos/calendar/coding/approvals happen on the web); cloud_strong → cloud_fast; short plain-text answers.
    #    The grounded graph can still ask its own clarifying question on LINE.
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

Model aliases are defined in **LiteLLM** `config/litellm.yaml` (the file is the source of truth; summary below);
the harness only calls `http://litellm:4000/v1` with `model=<alias>`. Model names come from `.env`
(`LOCAL_MODEL`, `CLOUD_FAST_MODEL`, `CLOUD_STRONG_MODEL`; see `.env.example`):
```yaml
model_list:
  - model_name: local          # GPU; keep_alive 10m
    litellm_params: { model: os.environ/LOCAL_MODEL, api_base: http://ollama:11434, keep_alive: 10m }  # ollama_chat/qwen3.5:9b (no q6_K tag upstream)
  - model_name: local_cpu      # same model, CPU only (not a separate q4_K_M tag)
    litellm_params: { model: os.environ/LOCAL_MODEL, api_base: http://ollama:11434, num_gpu: 0 }      # VERIFY num_gpu is forwarded
  - model_name: cloud_fast
    litellm_params: { model: os.environ/CLOUD_FAST_MODEL, api_key: os.environ/OPENAI_API_KEY }
  - model_name: cloud_strong
    litellm_params: { model: os.environ/CLOUD_STRONG_MODEL, api_key: os.environ/ANTHROPIC_API_KEY }
router_settings:
  fallbacks: [{ cloud_strong: [cloud_fast] }, { cloud_fast: [cloud_strong] }]   # never fall back to/from local here; privacy is decided upstream
  num_retries: 2
  timeout: 120
litellm_settings:
  drop_params: true
  callbacks: litellm_hooks.proxy_handler_instance   # kill-switch backstop (config/litellm_hooks.py, §19)
  max_budget: 10              # global daily USD hard cap (VERIFY)
  budget_duration: 1d
general_settings: { master_key: os.environ/LITELLM_MASTER_KEY }
```
Thinking (`think`) and `max_tokens` are set per call by the harness, not in LiteLLM (§12a `local_think_tasks`).
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

### Planned: review before approval (decided 2026-11-05, not built)
- After the harness tests pass and before the approval interrupt, run one more Copilot job in the same sandbox
  (`kind: review`, read-only tools, no writes to `/workspace`) that runs the code-review and security-review agents
  on the diff (non-interactive equivalents of `/review` and `/security-review`; exact invocation to VERIFY).
- The approval question/Inbox item gets the findings (severity, file:line, one line each) under the diffstat;
  high-severity findings are shown first. Findings never block — the owner still decides. A review error or timeout
  adds "review unavailable" and continues.
- Counts toward the same `--max-ai-credits` budget; skipped for docs-only diffs; `coding.review.enabled` toggle.
- Acceptance: a diff with a planted bug and a planted secret produces findings in the approval item; a clean diff
  produces "no issues"; the review job cannot modify the clone.

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
- **Verified live 2026-11-04:** a real Copilot CLI session posted hooks, was summarised into `coding_episodes` (19 turns), and `recall`/`get_thread_summary` returned it from inside Copilot. The local summariser can misname files (e.g. said `app/jobs/schedules.py` for `app/schedules.py`).

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
- Corpus: a read-only docs folder (`DOCS_DIR`, default `~/agent-docs`, mounted at `/docs`) plus uploads via `POST /api/agent/kb/documents` (stored in Postgres only). The vault (§10a) is a second root (`source = vault`).
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
  (`lscr.io/linuxserver/obsidian`, not verified — not built, see Implemented below), bound to LAN/VPN only.
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
2. Obsidian **Importer** plugin (official) → Notion → import zip into `vault/Notion/` (resolved 2026-11-04: it also imports via the Notion API; databases → folder + `.base`, primary view only — see Implemented below).
3. Post-process script: add frontmatter `source: notion`, `notion_id`, `privacy`; fix remaining links; convert database CSVs to notes or Bases; move files into `Attachments/`.
4. Validate: page count vs Notion, broken-link report, spot-check databases and embeds.
5. Index into RAG; keep Notion read-only for ~1 month, then archive.

### Implemented 2026-11-04 (phase 10a)

**Mounts & services** (`VAULT_DIR`, default `/home/hoder/vault`; `host/vault-init.sh` creates the folders, `.gitignore`, `.stignore`, `Inbox/README.md` and the git repo — run it before `compose up`, the bind mounts need `Inbox/` and `AI/`)
- Every app container: vault at `/vault` **read-only**, plus `Inbox/` and `AI/` at `/vault-rw/Inbox|AI` read-write. Nothing else in the vault is writable from a worker.
- `agent-api` only: the whole vault at `/vault-edit` read-write, used by exactly one code path, applying an owner-approved edit (Inbox approve).
- `vault-indexer` service (`python -m app.rag.watcher`, ro vault): full scan at start, then `watchfiles` → debounce (`vault.debounce_ms` 2 s) → `index_changed`. Ignores hidden files, Syncthing temp files and the writer's `.~harness-*` temp files. Heartbeat in Redis for the healthcheck. The 5-minute `index_docs_cron` also rescans the vault (safety net); a missing mount never deletes the index.
- `syncthing` service (`syncthing/syncthing:2`, uid 1000): volume at `/var/syncthing` (the image only chowns `$HOME`; config in `/var/syncthing/config`), vault at `/var/syncthing/vault`, folder id `vault`, staggered versioning, fs watcher on. GUI on host `127.0.0.1:8384` only (SSH tunnel); 22000 tcp/udp + 21027/udp for sync. The only service besides egress-proxy on `egress`.
- Host: `agent-vault-commit.timer` (00:15) runs `host/vault-commit.sh` → one `vault: YYYY-MM-DD (n files)` commit per day. `backup.sh` copies the vault (with `.git`) and the syncthing volume.

**Indexing** (`app/rag/vault.py`; chunks/documents with `source = 'vault'`, path = vault-relative)
- Frontmatter (`tags`, `aliases`, `privacy`), inline `#tags` (not in code), `[[wikilinks]]`, `![[embeds]]`, `[md](links.md)`. Tags and aliases are stored on the document and added to the keyword index; a query that equals a note's title or alias always keeps that note (exact "named" match, even below the rerank floor).
- Link resolution like Obsidian: exact path → path suffix → shallowest note of that name; relative md links resolved from the note. `note_links(src, dst, dst_path)` (dst unresolved until the note exists). Search seeds `rag.link_seeds` (3) top hits and adds up to `rag.link_candidates` (3) chunks of 1-hop linked notes, forward or backlink, before rerank.
- Rendered text for chunking: wikilinks → their display text, embeds → `📎 name`. Embedded attachments (`vault.attachment_kinds`: doc, image; ≤ 50 MB) go to the media-worker (stored with `private=True`, so local only); the extracted text is indexed under the embedding note as heading `📎 name` and cached by sha in `vault_files`. An attachment change reindexes the notes that embed it.
- Privacy: frontmatter, `privacy_rules.yaml` `private_paths` (vault-relative globs), Gate 0 content scan → `privacy_tag`. `AI/Daily/` is private only when that day includes a PRIVATE thread (§6, 2026-11-06).
- Citations: `[[Note#Heading]]` (vault path without `.md`, last heading); attachment text cites `[[file.pdf]]`.
- Measured: a file edit is searchable ~0.3 s after the write (watcher), well inside the 1-minute target.

**Writing** (`app/vault_writer.py`; agent tools in `app/tools/vault.py`)
- `capture_note(text, title?, tags?)` → `Inbox/YYYY-MM-DD HHmm title.md`; `append_daily_note(text)` → `## Log` of `AI/Daily/<today>.md`; `edit_note(path, old, new)` — exact single-occurrence replace. Assistant route only (with todos); `read_note` on knowledge routes. `/note <text>` saves to Inbox directly (works on LINE).
- Writes resolve under `/vault-rw/<Inbox|AI>` with a realpath containment check (symlinks out are refused) and an atomic temp+rename. Privacy only ever upgrades (a private thread or private text → `privacy: private`).
- `edit_note` outside `Inbox/`/`AI/` creates an Inbox approval item with a unified diff and `detail.vault_edit {path, old, new, base_sha}` (deduplicated; LINE push without content if `vault.notify_line`). Approve applies it through `/vault-edit` only if the file sha is unchanged (else 409, re-propose); dismiss = reject; audited (`vault_edit`).
- Nightly `vault_daily` schedule (23:50): `app/vault_daily.py` writes the summary block (between `<!-- harness:summary -->` markers, so the `## Log` survives) of that day's conversations, coding sessions, todos done and Inbox captures.

**Copilot / MCP**: `search_notes(query, k)` (vault only, never PRIVATE, redacted, wrapped `<untrusted>`) and `read_note(path)` (path traversal refused, PRIVATE notes refused) on the `harness-memory` MCP server.

**Notion** (verified 2026-11-04): the official Obsidian Importer supports both the Notion **API** (integration token `ntn_…`; databases become folders + a `.base` file, primary view only) and the HTML zip export (choose HTML, *Include everything*, *Create folders for subpages*; databases flattened). Import into `Notion/`, then `python -m app.tools.vault_tool notion-fix [--strip-ids] [--dry-run]` (md links → wikilinks, frontmatter `source: notion` + `notion_id`, optional removal of the 32-hex id from file names with link rewrite) and `vault_tool links --folder Notion` (broken-link report, exit 1 if any; `--all` also counts wikilinks to notes not written yet).

**Verified live 2026-11-04**: phone-style edit → indexed in 0.3 s; tags/aliases/links stored; alias query hit; 1-hop expansion both directions; MCP tools over HTTP; worker has no write access outside `Inbox/`/`AI/` (read-only FS error on `/vault`, no `/vault-edit`); `/note`, capture and daily writes owned by uid 1000; an edit to `Notes/` became an Inbox diff and applied once on approve (second approve refused); nightly daily note + vault commit; PDF/text embed extracted by media-worker and cited. `rag_eval` is pinned to `source: docs` per case (default), unchanged at recall@5 1.00 / 0.88.

**Not done / follow-ups**: browser Obsidian container (optional; Obsidian desktop/mobile via Syncthing instead); §12 verify-mode `kb_search`/`kb_read` for PRIVATE threads still return "local KB not available" (cite_only mode already cites the vault); Notion import itself is a manual owner step.

## 10c. Knowledge curation — planned (decided 2026-11-05, not built)

Goal: the vault fills up with useful, well-described notes without the owner writing them from scratch. Notion
notes were written tersely for the owner's own eyes and retrieve poorly; repeated work lives only in short todos.
A periodic job drafts notes and edits from what the harness already sees; **nothing lands outside `AI/` without
owner approval**.

**Where content belongs** (applies to the Notion import and to curation output)
- **Content vs views:** content has one home; views only reference it. Notes → vault (`Notes/`, `Projects/`,
  `Resources/`); open todos → `todos` table (nodeReact `/Todos`, reminders, tools, MCP); events → Google Calendar;
  short facts/preferences → `/remember` memories; secrets → a password manager, never the vault.
- **Views** (dashboards, Bases, boards, link lists) go in `Views/` and are excluded from RAG — they carry little
  text and crowd out real notes. Needs a new `vault.exclude_paths` (vault-relative globs) in the indexer.
- **Worth keeping as notes** ("would I ask the agent this in six months, and only I know the answer?"): decisions
  and their reasons, problems and fixes, verified procedures (SOPs), project context, learning summaries in the
  owner's words, personal reference (models, specs, account purposes, contacts), meeting outcomes. Not worth it:
  copied web articles (keep only the owner's comments + link), stale temporary info, done todos (one archive note).
- **Repeated work:** each occurrence is a todo; *how to do it* is one SOP note (`Resources/SOP/<name>.md`: steps,
  checklist, systems/files, pitfalls, typical duration) that grows each time; a one-line preference is a
  `procedures` memory. Todos have no recurrence field; a schedule can create them (verify that a `prompt` schedule
  turn can call `add_todo`, else add a schedule kind that creates a todo linking its SOP).
- **Terse imported notes:** import as-is (`notion-fix`, `links --folder Notion`), measure with `rag_eval` cases
  pinned to `source: vault`, enrich only notes that miss. Enrichment is additive: better titles, `aliases`/`tags`
  (indexed as keywords; an exact title/alias match is always kept), `[[links]]` (1-hop expansion), and a short
  context block at the top of the body. The folder path and other frontmatter fields (e.g. `summary:`) are **not**
  indexed — embedding text is `title + heading + text` (`rag/store.py`), so context must be in the body.
  Notes that don't fit "worth keeping" go to `Archive/` instead of being enriched.

**Curation job** (schedule kind `curate`, weekly, background queue)
1. **Sources** since the last run: done todos (title + notes), episode summaries, `Inbox/` captures, `AI/Daily/`
   notes, Copilot session episodes, new memories, and existing notes that failed retrieval (eval misses).
2. **Group:** embed items and cluster; a cluster becomes a candidate when it recurs (e.g. ≥ 3 similar done todos)
   or matches an existing note.
3. **Proposal kinds**, ranked by value, **at most `curate.max_proposals` (5) per run**:
   - **new note** — SOP, decision record, problem/fix note; draft written to `AI/Curation/`;
   - **update** — new facts appended to an existing note (diff);
   - **tidy** — add aliases/tags/`[[links]]`, merge duplicates, file `Inbox/` captures into a folder;
   - **archive** — stale or unused notes → `Archive/`.
4. **Missing facts are asked, not invented:** a draft lists open questions (e.g. "which system is used in step 3?")
   in a marked block; the owner's answers in the Inbox item are folded into the final note.
5. **Approval** through the Inbox (`kind: approval`, `detail.curation`):
   - update/tidy/archive reuse the `vault_edit` flow (unified diff, `base_sha` check, 409 → re-propose);
   - new notes need a new action: approve = move the draft from `AI/Curation/` to the owner-chosen folder via
     `/vault-edit` (the model suggests a default folder); reject = delete the draft;
   - rejections are recorded (cluster hash + kind) and not proposed again; dismissing every proposal of a run
     lowers the next run's cap.
6. Generated text is wrapped in `<!-- harness:curated -->` markers so the owner's own text stays distinguishable.

**Routing & privacy** — curation runs through the normal pipeline (Gate 0 → router → `decide()`), not a side path
- The job is a normal `Job` (channel `cron`, background queue), so budget, kill switch, GPU state and the decision
  log apply unchanged.
- **Gate 0 per source item, batches split by privacy:** one PRIVATE item makes a whole prompt PRIVATE, so items are
  gated individually and grouped into PRIVATE and non-PRIVATE batches.
  - PRIVATE batch (owner-marked only, per the revised privacy model in §6: frontmatter `privacy: private`,
    `private_paths`, private threads' episodes) → local only (hard rule, `decision.py` rule 2).
  - Non-PRIVATE batch (most material, including `AI/Daily/`) → the router may pick `cloud_fast`/`cloud_strong`;
    the model only ever sees redacted text.
- **Reversible pseudonymisation (new):** today `gate0.redact` replaces matches with `[REDACTED:<name>]`, which is
  one-way. For curation, replace each match with a stable placeholder (`<EMAIL_1>`, `<PHONE_2>`, …), keep the map
  locally (Redis, per job, TTL), and restore the originals in the cloud draft **locally** before it reaches the
  Inbox. The map never leaves the harness; a draft with an unknown placeholder is rejected.
- Names, companies and addresses are not redacted (accepted under the revised privacy model); no Gate 1 needed.

**Read-only note viewer (nodeReact)** — for "just show me the note" without a model call
- Today the chat answers from the top RAG chunks (a paraphrase, model cost, a few seconds) and citations such as
  `[[Note#Heading]]` are plain text; `read_note` shows a whole note only when asked explicitly.
- API: `GET /api/agent/notes/{path}` (owner token, web only) → `{path, title, frontmatter, body, privacy_tag,
  mtime, links, backlinks}` read from `/vault` (read-only). Path checks like the MCP `read_note` (vault-relative,
  realpath containment, no hidden files); `Views/` and PRIVATE notes are allowed (owner, own UI). Never exposed on
  LINE or to unauthenticated routes. Attachments via `GET /api/agent/notes/{path}/attachment?name=` (images
  inline, others as download; same containment check).
- nodeReact page `/Notes?path=…#heading`: renders Markdown (wikilinks and embeds resolved like the indexer,
  `[[x]]` links open the viewer, a missing target shows as unresolved), scrolls to the heading, lists backlinks.
  No editing — edits stay in Obsidian (or the Inbox approval flow).
- Chat: `[[Note#Heading]]` citations in answers and `done.grounding.citations` for `source: vault` become links to
  the viewer; docs-folder citations (`path#heading`) open the same viewer if the file is under `DOCS_ROOT`.
- Optional: a simple title/alias search box on `/Notes` backed by `kb/search` (no model).

**Acceptance**
- The note viewer opens a cited note from a chat answer in one click, without a model call; path traversal and
  hidden files are refused; nothing is writable through it.
- A run produces ≤ 5 proposals; nothing is written outside `AI/` before approval; approve, reject and stale-sha
  paths work for every proposal kind.
- A PRIVATE source item never reaches a cloud model (decision log + LiteLLM logs); cloud drafts contain no raw Gate 0
  findings, and placeholders are restored before the Inbox.
- A rejected proposal is not proposed again; `Views/` notes never appear in `search_knowledge` results.

## 10d. Todo categories & personal logs — planned (draft, decided 2026-11-05; details open)

**Todo categories** (from Notion: separate pages today)
- Add `category` to `todos` (free text with autocomplete; maybe `tags` later). Initial: work, side project,
  personal, shopping. `personal` may be a PRIVATE category (local only, no LINE content).
- `/Todos`: grouped by category with counts, category filter, a "Today" view (overdue + due today across
  categories). Shopping renders as a compact checklist without due dates. `add_todo` (tools, MCP) gets `category`.
- Import: Notion CSV, open items only, category per source page.
- Overdue todos are never deleted or auto-closed: they stay open and are marked overdue (red "逾期 N 天" badge in
  `/Todos` and the Today page, an "overdue" group at the top, `list_todos` output and the morning brief say
  "overdue"). Only the owner completes or deletes a todo. (Today nothing auto-deletes either; the change is the
  overdue display.)

**Personal logs** (reviews live in the vault, not in `todos`)
- Media (book, movie, anime, series, game) share one collection `Media/` with `type`; trips, food and product
  reviews get their own collections. Shared frontmatter: `type`, `status`, `rating`, `date`, `tags`.
- One note per item: the owner's own words kept verbatim; AI-written sections inside `<!-- harness:curated -->`.
- Review flow: owner sends "title + raw feedback" (e.g. the 2026-11-05 *Starman* chat) → AI fills a per-type
  template → preview → saved on confirm. Factual fields (year, cast, IDs) only from pages actually read, else
  marked unverified; the AI never invents a rating.
- Agent may create notes in these collections and update `status`/`rating` without approval; body edits still go
  through Inbox approval.
- nodeReact `/Logs`: read-only list by type and status, quick add and status/rating edits (frontmatter only), links
  to the note viewer (§10c) and `obsidian://`. No review editor.
- Obsidian Bases views in `Views/` per collection, plus a cross-type "all reviews" view.

**Open:** folder names and per-type templates/fields, note language (fixed or follow the input), exact write
permissions for the agent, whether `personal` is always PRIVATE, `tags` on todos.

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

> **Status:** implemented 2026-11-04 (phase 10b): `app/verify.py`, called by `jobs/worker.run_agent` before `done` for every grounded/ReAct answer (coding keeps its own test-based verify). Policy block `verify:`.

Verify (skip for trivial chat):
1. Deterministic: coding → harness-run tests/lint/build; assistant → schema valid, no tool errors, citations present when RAG used.
2. Model check: Jev `Noul("The response fully and correctly answers the request")` for non-private; Qwen for private; deterministic-only when GAMING + private.
3. Fail → escalate tier (local → cloud_fast → cloud_strong), max 2 escalations → Inbox item.

Implementation notes (10b):
- Skipped: coding, the ReAct `clarify` route, interrupted runs, and `chitchat` answers under 300 chars.
- Deterministic (no model, checked first): empty answer (grounded: `answer_body`, without the 「我理解為」 line), ≥3 distinct Simplified-only characters (not for shopping lists: store titles), a tool error in this turn, the required write tool (`tools.required_tool`) not called. "Sources read but none cited" (`require_citations`) is off: live, all three tiers failed it on a correct answer because the search had found the wrong sources, and a bigger model reruns the same retrieval.
- Model check: Jev Noul above on redacted request + answer (~200 ms); local Qwen JSON score when PRIVATE or the kill switch blocks cloud; none when that would need the GPU while GAMING. Fail below `min_score` 0.3 (calibrated on 77 real turns: every answer ≤0.3 was wrong, empty or off-topic). A checker error or timeout passes.
- Escalation: the worker records where the turn started (`runmeta.verify_start`: pre-turn checkpoint, or the clarify interrupt for a resumed turn) and requeues the same run_id on the next tier, which forks the thread from that checkpoint (a resumed turn: `aupdate_state` + `Command(resume)` with the owner's answer). The failed answer drops out of the history; the re-run never asks a clarifying question again (`no_clarify`). Web gets `reset` + `status: requeued`. Never escalated: PRIVATE, kill switch cloud/all, a turn that called a write tool (todo/event/remember would run twice).
- Final failure: the answer is delivered with a ⚠️ warning line (web and LINE) and an Inbox item `kind=verify` (request, answer, checks, models tried). `decisions.outcome.verify` records every tier's result; `cost_usd` is the total of all tiers.
- Cost: cloud_fast averages $0.0013 per run, cloud_strong $0.034. At the calibrated ~25% failure rate that is ≈$0.0003 per turn for the first escalation, plus cloud_strong for the turns that fail again (the second step is ~25× the first; set `max_escalations: 1` to stop at cloud_fast).

Approval:
- `interrupt()` before: git push/PR merge, send mail/message to others, delete, calendar create/delete, any money action, network enablement in sandbox, any action flagged `risky`.
- Background approvals persist (Postgres checkpoint) → Inbox card in nodeReact + LINE push notification → resume on approval.

Output: SSE (web), LINE reply/push (never PRIVATE content), Inbox, memory write, PR link.

### Planned: plan critique (decided 2026-11-05, not built)
The L9 check above asks "does the answer address the request". Plans need a second opinion on *content*:
- **Applies to** generated plans: travel itineraries, shopping recommendations, SOP/curation drafts (§10c), review
  drafts (§10d), and other multi-step plans (task type / route marked `critique: true` in policy).
- **Critic:** a different, stronger model than the author (default `cloud_strong`; local Qwen on PRIVATE threads)
  with a per-type checklist, e.g. itinerary → time feasibility, travel times, opening days, budget, missing
  bookings; shopping → budget, real listings, requirement fit; SOP → missing steps, unclear owner, invented facts.
- **Output:** a structured list of issues (severity, what, suggestion). High-severity issues trigger one revision
  by the author model; remaining issues are appended to the answer as a short "⚠️ Check" block. Max one revision.
- Runs after the L9 check, before `done`; counts toward the turn's cost cap; when the kill switch blocks cloud the
  critic falls back to local Qwen, and it is skipped when that would need the GPU while GAMING.
- Acceptance: an itinerary with an impossible transfer or a closed-day visit is flagged; a clean plan passes with
  no block; at most one extra revision per turn.

### Planned: per-type checks beyond L9 (decided 2026-11-06, not built)
L9 stays for every type. These add checks that fit each type, mostly in code (no extra model call):

| Type | Extra checks | On failure |
|---|---|---|
| chat (§12d) | citation check (today) | L9 escalation; on LINE warning only (§19) |
| research (§12d) | numbers/dates/prices in the answer must appear in a cited excerpt (code match); time-sensitive questions flag sources older than N days; conflicting sources are shown side by side, not resolved silently; claim checks run on a cloud model (not PRIVATE) | unmatched claims go to "Unverified"; next-step suggestion to confirm from a primary source |
| deep research (§12d) | plan critique of the sub-questions before starting; per-section claim checks as research; key conclusions need 2 independent domains or 1 T1 source; spot-check quotes by re-fetching the page; final checklist review by a different model (answers the question, gaps, contradictions) | issues listed at the end of the report |
| assistant | **claimed action without a tool call** ("已加入/已建立" but no write tool ran this turn) fails; **read back** after a write (todo/event exists with the requested date, time and title); relative dates ("明天下午三點") resolved in code and echoed with the weekday ("11/7 (五) 15:00"); warn on a likely duplicate todo/event | claimed-action failure reruns once without escalation after writes; read-back mismatch is reported in the answer |
| shopping (§12c) | every pick within budget (code); required specs (capacity, size, model) present in the listing title/specs; show price fetch time and re-fetch top picks when older than N hours; one entry per product across stores (or merged); dead links dropped | failing picks are dropped or labelled |

Priority: assistant claimed-action and read-back first (cheapest, most common failure), then shopping budget/spec,
then research number matching.
**Open:** N for freshness (research, shopping), duplicate rules for todos/events, deep-research quote sample size.

## 12a. Grounded answering (clarify → verify claims → cite) — decided 2026-10-22, implemented 2026-10-28 (phase 8a; default mode cite_only)

Owner preferences, applied to **every** answer:
1. **Clarify first** whenever intent/scope is uncertain. *Intent* uncertainty → ask the user; *factual* uncertainty → search (never ask "should I search?").
2. **One authoritative, primary source that directly supports each factual claim.** Secondary sources are never cited.
3. **Insufficient evidence → partial answer** with unverified content explicitly labelled.

Claims are classified first: greetings, creative writing, opinion, pure reasoning/math are `exempt` (no source); every factual claim needs a citation or an "unverified" label.
PRIVATE threads never use web tools; they may cite only the local knowledge base (§10a). In the default `cite_only` mode they cite the docs folder and vault (phase 10/10a); in verify mode every factual claim on a PRIVATE thread is still labelled unverified (follow-up, see below).

### MVP subset (implemented 2026-10-22)
The full graph below is phase 8a. The MVP ReAct graph already does:
- **Clarify on local, escalate after:** decision rule 3 (§7) sends ambiguous requests (intent confidence < `clarify_below` 0.7 (0.8 until 2026-11-03), or Jev `ambiguous` > `thresholds.ambiguous` 0.7 — intent confidence alone misses vague referents like 「那個東西多少錢」, which scored 0.98) to `route=clarify` on `local` (`local_cpu` when the GPU is BUSY/GAMING). Redis `clarify:{thread_id}` (`{text, redacted_text, question, rounds}`, TTL `clarify.context_ttl_s`) makes the next message route as `request + [Assistant asked] … + [User answered] …`; the router's tier on that text decides any escalation. On PRIVATE threads the question is left out of the redacted copy. `clarify.max_rounds` (10, raised from 5 on 2026-10-28) → answer with stated assumptions. `clarify_rounds` is logged in `decisions.inputs` for replay.
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
- PRIVATE: `kb_search()` / `kb_read()` with the same interfaces (`domain = vault:<path>`); they still return `error="local KB not available"` (follow-up: back them with `store.search(private=True)` now that §10a has shipped). (Phase 10: in the default `cite_only` mode, private threads already cite the owner's KB from §10, with no web; verify mode below is unchanged.)

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
  local_think_tasks: [compose]      # Qwen thinking per step on local (see Implementation notes)
  local_think_tokens: 4096
executor:
  recursion_limit: 60               # max_rounds (10) clarify ×2 + 3 research loops ×5 + fixed nodes ≈ 45
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
  - **`grounding.always_search: true` (owner decision 2026-11-03):** `needs_web` is ignored and every non-PRIVATE turn
    searches, including reviews, rewrites and advice about a named subject (the 9B model's own casts, links and dates
    were often wrong, and it judged such requests "creative writing"). Skipped only for `search_skip_task_types`
    (`chitchat`) and when the owner's documents returned a hit ≥ `rag.skip_clarify_score`. With no planned query,
    the planner's `entity` is searched.
  - **Taiwan first (owner decision 2026-11-03):** the planner and `BASE_SYSTEM` assume Taiwan for place-dependent
    questions and write Chinese queries in Traditional Chinese. Brave gets `country=TW` (`grounding.search_country`) and,
    for Chinese queries, `search_lang=zh-hant` (`search_lang_cjk`); both search tools. Hits are fetched in the order
    T1 → Taiwanese (`authority.yaml region.first_*`, `.tw`) → other → mainland-China (`region.last_*`, `.cn`, 163.com…),
    then by tier. Before this, 「今天天氣好嗎」 searched 「中国今日天气」 and read sina/163 pages.
  - It then searches, ranks hits as above (rules only; rejected domains are dropped) and fetches up to `cite_pages` pages of `cite_chars` each.
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
  - On PRIVATE threads, every factual claim is labelled unverified (`private_kb`) until `kb_search` is wired (follow-up after §10a).
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
  - `grounding.local_think_tasks` re-enables thinking per step with `local_think_tokens` extra headroom. [verify_support, compose] measured about 140 s per turn (over `max_active_s`) with empty answers; since 2026-10-29 only `[compose]` (4096). If thinking still uses the whole budget and leaves no visible text (seen 2026-11-03: 5.7k tokens, LINE got only the 「我理解為」 header), `call()` retries once with `think: false`.
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
| Clarify | ambiguous referent → one interrupt with `slot=referent`; clear question → no interrupt; `route=clarify` asks when the model has a real question (no "null"/generic question, 2026-11-03); `clarify.max_rounds` ambiguous rounds → no further interrupt, answer has assumptions; web resume merges answer into `question`; LINE next message resumes, Gate 0 runs, `/skip` works; PRIVATE clarify sends notice only on LINE; waiting 10 min (fake clock) not counted in `active_ms` |
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

**Goal:** "幫我推薦調味料罐" should end with 3–5 real products the owner can buy. Each one shows its real price, store, rating and link. Before that, the bot asks for whatever is missing (item and budget). It is a branch of the grounded graph (§12a), so the web clarify UI works unchanged.

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

## 12d. Answer tiers — chat / research / deep research — planned (decided 2026-11-05, not built)

Three depths instead of one global `grounding.mode`; the tier is chosen per request.

| | **chat** | **research** | **deep research** |
|---|---|---|---|
| Based on | today's `cite_only` | today's `verify` mode (§12a) | new background job |
| Web | search + read; only citations checked | search + read; every claim checked, "Unverified" block | multi-round search, sub-questions, cross-checked sources |
| Clarify first | only when very ambiguous | when scope is unclear | confirms scope and budget before starting |
| Runs | foreground, seconds (today's caps: 3 searches, 4 reads, 90 s, $0.05) | foreground with progress, up to ~2–3 min, own caps (`research.*`) | background, minutes to tens of minutes, own budget (e.g. ≤ $1) |
| Output | chat answer | chat answer + verification result + next steps | report saved to the vault (`AI/Research/`) + Inbox item; follow-ups in the thread read the report |
| Model | router decides | cloud (verify mode is too noisy on the 9B model) | `cloud_strong` |

**Choosing the tier**
- Explicit prefixes `/chat`, `/research`, `/deep` (like `/local`, `/code`).
- Otherwise the Jev router picks **chat vs research** only (both foreground, so a wrong pick costs seconds, not
  minutes). Deep research is never automatic: only `/deep` or a next-step suggestion the owner picks.
- Router choice and owner overrides are logged in `decisions`, so the chat/research split can be tuned with replay.

**Next-step suggestions** (how an answer leads deeper; generated in the same model call, negligible cost)
- research answers end with 2–4 suggested next steps derived from the result: confirm "Unverified" claims from a
  primary source, compare options the answer mentioned, expand a sub-topic, or escalate to deep research.
- Each suggestion shows its tier and rough cost/time (e.g. "🔍 research", "📚 deep research, ~10 min"); web shows
  buttons (reuse the clarify option UI), LINE quick replies. Picking one runs it as the next turn in the same
  thread with that tier, so earlier sources are reused rather than re-fetched.
- chat: 2–3 follow-up questions in the same message as the answer (web buttons, LINE quick replies on that
  message, never a separate message). Add "research this?" only when the router saw a research-type intent.
- deep research: the report ends with "open questions", each one startable as a new research/deep run.
- The system never escalates on its own; every step deeper is the owner's choice.

**Chat speed (decided 2026-11-06, not built)**: the chat tier is web-first and should answer quickly, but it does
not need to be extremely fast. Routing and model choice stay as they are (Jev picks the tier, about 80 % of web chat
runs on the local 9B model).
- Verification: chat keeps today's L9 whole-answer check (§12, phase 10b: score, rerun on a bigger model,
  warning). It has no per-claim check; that is the research tier.
- Always search: already on (`grounding.always_search: true`, chitchat skipped). The `plan_search` model call that
  writes the queries stays.
- Parallel queries: keep the current count (at most 2 queries from `plan_search`, cap `tool_limits.web_search: 3`).
  Ask for different angles (e.g. official/primary source + latest news or reviews). Run them concurrently instead
  of one after another (`grounded.py` cite_only search loop). Page reads run concurrently too.
- Snippets first: give the compose step the Brave result snippets. Read full pages only when the snippets are not
  enough. Each page read gets a short timeout (about 2–3 s); a page that is too slow is skipped, not waited for.
- One message per turn: no progress messages, partial answers, or late follow-up messages (LINE replies are
  limited, so every extra message costs). The web client keeps its existing streaming and status events.
- Measure: log time to first token and total turn time per answer, so the change can be checked.
- Done 2026-11-06: `plan_search` sees the last `grounding.search_history_turns` (3) earlier turns, so follow-ups
  like "那價格呢?" get the subject in the query; the assistant route keeps 20 turns (was 8).
- LINE threads are chat tier only (§19 "Planned: notifications on Telegram, LINE replies only").

**PRIVATE threads:** local models only (§6), so research quality drops; research on a PRIVATE thread may fall back
to chat with a note, and deep research is refused unless the owner confirms a local-only run.

**Open:** research/deep caps, report format, how many deep runs may run at once, whether deep research uses
several parallel workers (multi-agent collaboration discussed, not decided), how chat decides that the search
snippets are enough to skip reading full pages.

## 12e. Tutor agent — planned (decided 2026-11-06, not built)

A language tutor, English first (owner level B2; goals: travel and reading), later Japanese and Spanish, later other
subjects. Web only.

**Entry and routing**
- Own route `tutor`, entered with `/tutor` or from a dedicated tutor thread on the `/Tutor` page. Jev never routes
  there on its own. Model choice follows the normal rules (Jev picks the tier; L9 escalates on failure).
- One **subject profile** per subject (`config/tutor/<subject>.yaml`): level, goals, focus, exercise types, checks.
  A new language or field = a new profile, no new code path.

**Learner profile** (long-term memory, per subject): level, goals, recurring error types. Every correction updates
the error log; the most frequent error types drive the next exercises and grammar lessons.

**Exercise types (MVP: all five)**
- **Conversation + correction**: free talk or travel role-play (hotel check-in, ordering, directions, transfers,
  lost luggage, booking problems, feeling unwell); the agent plays the other side. Corrections at B2 focus on
  naturalness, collocations, prepositions, register and word choice; at most 2–3 main points per reply, the full
  correction in a collapsible block.
- **Writing correction**: travel writing (emails to hotels, complaints, reviews) and reading summaries; sentence by
  sentence: original → fixed → one-line reason.
- **Speaking (voice input)**: existing ASR with the language fixed to English; the raw transcript is shown. Whisper
  tidies speech (adds articles, drops fillers), so spoken grammar errors may be hidden and pronunciation is not
  assessed; this mode trains fluency and sentence building only.
- **Vocabulary cards (SRS)**: words and collocations proposed from corrections and readings; the owner ticks which
  to add (daily new-card cap, e.g. 10). FSRS scheduling, grades again/hard/good/easy. A card holds the item,
  meaning, the owner's original sentence and the fix, one example, and its source (e.g. a book from §10d).
  New table `tutor_cards` (+ review log); not in the vault.
- **Grammar lessons**: offered when one error type reaches a threshold (e.g. 3 times): a short explanation plus 3
  exercises graded in code.

**Reading**: the owner brings an article or URL (or asks for one at B2–C1 on an interest) → 3–5 comprehension
questions, word/phrase picks for cards, a written summary that is corrected. Books from the reading list (§10d)
tag their cards with the title.

**Travel tie-in**: when an upcoming trip is in the calendar or vault, the tutor may suggest role-plays for it.

**nodeReact `/Tutor`** (tabs): conversation (corrections in a side panel), writing, review (card flip + grades),
progress (error types over time, errors per 100 words, card retention). The Today page shows "N cards due"; the
daily review reminder goes through Telegram (§19), never LINE.

**Checks** (in addition to L9, §12): a correction must quote the owner's original text and give both versions;
card and word explanations must have every required field (meaning, part of speech, example + translation),
checked in code. For Japanese later: turn off the L9 Simplified-Chinese check on the tutor route (Japanese kanji
such as 学/会/国 overlap with Simplified). Watch the L9 failure rate on the 9B model; revisit model choice only if
escalations are frequent.

**Open:** card table schema and FSRS library, error-type taxonomy, whether role-plays draw on real trip data
automatically or only on request, Japanese/Spanish profiles (furigana, JLPT/DELE levels).

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
    build: ./media                       # ffmpeg, faster-whisper (ctranslate2), Docling, RapidOCR, opencc
    command: arq media.worker.Settings   # queue arq:media
    volumes: [media-models:/models, /srv/agent/files:/files:ro]
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
    image: syncthing/syncthing:2
    volumes: ["/home/hoder/vault:/var/syncthing/vault", "syncthing:/var/syncthing"]   # image chowns only $HOME
    ports: ["22000:22000/tcp", "22000:22000/udp", "21027:21027/udp", "127.0.0.1:8384:8384"]
    networks: [egress]
  vault-indexer:                         # python -m app.rag.watcher (§10a)
  # every app container additionally mounts:
  #   /home/hoder/vault:/vault:ro                 (indexer, read_note)
  #   /home/hoder/vault/Inbox:/vault-rw/Inbox     (writer)
  #   /home/hoder/vault/AI:/vault-rw/AI           (writer)
  # agent-api only: /home/hoder/vault:/vault-edit (applies owner-approved edits)
  # agent-api, agent-worker, media-worker mount /srv/agent/files:/files
  # agent-api, agent-worker, media-worker, sandboxes set:
  #   HTTPS_PROXY=http://egress-proxy:4750  NO_PROXY=litellm,ollama,agent-postgres,agent-redis,phoenix
  #   OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006
  # => remove `egress` from every service except egress-proxy and syncthing.
volumes: { ollama: {}, pg: {}, redis: {}, hf-cache: {}, media-models: {}, syncthing: {}, phoenix: {} }
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
| 1 | Compose skeleton: postgres, redis, ollama, **litellm**, **phoenix**, **egress-proxy**, agent-api; one ReAct chat graph; SSE; minimal React chat page in nodeReact — **implemented 2026-10-21** (MVP) | Chat streams tokens; thread survives agent-api restart; trace visible in Phoenix; non-allowlisted domain blocked |
| 2 | Gate 0 + Jev router + decision layer (+ Qwen fallback router; Gate 1 stub only); `decisions` logging + replay tool — **implemented 2026-10-21** (MVP) | Unit tests for `decide()` cover every rule; Jev outage → fallback works |
| 3 | Job layer (arq queues, cancel, dedupe) — chat already goes through it; **kill switch**; **ZFS backups** — **implemented 2026-10-21** (MVP) | Cancel button stops a run within one step; duplicate webhook ignored; `/stop cloud` blocks cloud calls; nightly backup + snapshot created |
| 4 | gpu-governor + Ollama unload — **implemented 2026-10-28, live-verified 2026-11-03** (§11: real Moonlight sessions entered GAMING via NVENC on 11-02; forced GAMING unloaded Qwen + embeddings 8.4 → 0.4 GB in ~1 s and a web turn routed to `cloud_fast` by rule 5, embeddings on CPU) | Starting `steam-headless` + streaming frees Qwen VRAM within 10 s; jobs route to cloud |
| 5 | Coding route: sandbox-manager + Copilot (headless, not ACP) + verify + approval — **implemented 2026-10-28** (§9a; verified end-to-end, squid gate passed) | Task produces diff, harness-run tests pass, push blocked until approved (nothing is ever pushed; approval creates a local branch) |
| 6 | Copilot ingest hooks + harness-memory MCP — **implemented 2026-10-30** (§9b; verified live 2026-11-04: hooks → ingest → `copilot:<id>` episode; `recall`/`get_thread_summary` from a Copilot CLI session) | SSH session appears as `copilot:<id>` thread; recall works from SSH Copilot |
| 7 | Todos + Inbox + background approvals — **implemented 2026-10-30** (+ 7b Google Calendar, Inbox-approved writes) | Todo CRUD from web, agent, Copilot; paused job resumes after approval days later |
| 8 | LINE channel + **health alerts** — **implemented 2026-10-21** (MVP; LINE is chat + notify only since 2026-10-30) | Owner-only; quick reply + push for long jobs; no PRIVATE content sent; stopping a container triggers one LINE alert |
| 8a | Grounded answering (§12a): clarify interrupt/resume (web + LINE), claim extraction, authority tiers, per-turn caps — **implemented 2026-10-28** (web verified live; LINE resume verified live 2026-10-28/10-30/11-03) | §12a test table passes; ambiguous question asks before answering; every factual claim has one T1 citation or an "Unverified" label; caps never exceeded |
| 8b-read | Browser rendering fallback (§12b stage 1) — **implemented 2026-10-28** | JS-only page returns text via `browser`; private/loopback/metadata URLs blocked by squid; normal pages not rendered |
| 8b-login | Interactive browsing + owner login (§12b stage 2) — planned | §12b stage 2 acceptance list |
| 8c | Shopping: clarify item/budget, recommend real store listings (§12c) — **implemented 2026-10-28** (web only; LINE is chat-only, no shopping branch) | A missing budget is asked with option buttons; every pick is a real listing with code-rendered price/store/link; 「cheaper」 lowers the ceiling; unrelated listings are filtered |
| 9 | Media: voice, image, doc, then video — **implemented 2026-11-04**, live-verified (§5: image, voice, docx, scanned + 25-page PDF, video in background; GAMING → CPU int8 ASR and OCR-only images; nodeReact attach button) | Each modality produces text; video runs as background job; CPU path works while GAMING |
| 10 | RAG + long-term memory + eval set + **memory review UI** — **implemented 2026-10-29** (§10; recall@5 = 1.00 reranked / 0.88 RRF-only on 33 questions; nodeReact `/Memory` page) | recall@5 measured on eval set; memories editable/deletable in nodeReact |
| 10a | Obsidian vault: Syncthing, indexer, writer tools, Notion import — **implemented 2026-11-04**, live-verified (§10a: edit → indexed in 0.3 s; writes only via `/vault-rw/Inbox|AI`, other edits by Inbox approval; Notion import tooling ready, import itself pending) | Notion pages imported with no broken links; edits on phone re-indexed within 1 min; agent writes only to `Inbox/` and `AI/` |
| 10b | General L9 verify layer (§12): deterministic checks + model check (Jev/Qwen) + tier escalation (max 2) → Inbox item — **implemented 2026-11-04**, live-verified (pass: Jev 0.86 in 206 ms; forced failure: local → cloud_fast → cloud_strong → Inbox, history keeps only the last answer) | A failed check escalates local → cloud_fast → cloud_strong at most twice, then creates an Inbox item; private runs never use cloud; extra model cost measured on the eval set |
| 10c | Knowledge curation (§10c): `vault.exclude_paths`, weekly `curate` schedule, Inbox proposals (new note / update / tidy / archive), per-item Gate 0 batching, reversible pseudonymisation; read-only nodeReact note viewer with clickable citations — **planned** | §10c acceptance list |
| 10d | Todo categories + personal logs (§10d): `todos.category`, grouped `/Todos`; review collections in the vault, template-based review flow, nodeReact `/Logs` — **planned (draft)** | to be defined |
| 11 | Cron/events, eval suite in CI, monthly restore test — **implemented 2026-11-04**, live-verified (brief → web thread + Inbox + LINE notice; restore test OK in 4 s) | Morning brief runs; eval regression check; restore test passes |
| 12 | Today home page in nodeReact (§19): calendar, todos, shopping, Inbox, in-progress logs; harness status shown only on exceptions (no separate agent dashboard); one aggregate endpoint, read-only — **planned** | §19 "Today home page" acceptance list |
| 13 | Finance dashboard (§19): Shioaji pull, finance-mailbox email ingest (statements, e-invoices) → `transactions`, owner-only nodeReact page — **planned (draft)** | to be defined |

**Remaining (2026-11-04):**
- **8b-login** — interactive browsing + owner login (§12b stage 2): not built.
- **PRIVATE verify mode** — back `kb_search`/`kb_read` (§12) with `store.search(private=True)`; until then verify-mode claims on PRIVATE threads are labelled unverified (`cite_only`, the default, already cites the docs folder and vault).
- **Gate 1** (§6) — deferred by design; build only if the decision log shows Gate 0 missing private data. Not needed under the revised privacy model (§6, 2026-11-05).
- **10d Todo categories & personal logs** (§10d) — draft; open questions listed there.
- **13 Finance dashboard** (§19) — draft; open questions listed there.
- **Answer tiers** (§12d) — planned: chat / research / deep research per request, Jev picks chat vs research, next-step suggestions lead deeper; chat speed (parallel queries, snippets first, one message per turn).
- **Telegram notifications + LINE replies only** (§19) — planned: every proactive message via one `notify()` to Telegram; LINE answers within the reply window, no push.
- **Tutor agent** (§12e) — planned: `/tutor` route, English (B2, travel + reading) first; conversation/writing/speaking correction, SRS cards, grammar lessons; nodeReact `/Tutor`.
- **Plan critique** (§12) — planned: stronger-model checklist review of itineraries, shopping picks, SOP/review drafts; one revision max.
- **Per-type checks beyond L9** (§12) — planned: assistant claimed-action + read-back first, then shopping budget/spec, research number matching, deep-research cross-checks.
- **Coding review before approval** (§9a) — planned: Copilot code-review + security-review on the diff, findings in the approval item.
- **12 Today home page** (§19) — planned: nodeReact `/Today` backed by `GET /api/agent/today`; harness health only when something needs attention.
- **10c Knowledge curation** (§10c) — planned: `Views/` exclusion, weekly curation proposals via Inbox, privacy-split routing, read-only note viewer (`/Notes`) with clickable citations.
- **Media gaps** (§5) — no cloud ASR/vision fallback; HEIC untested live; PDFs > 50 pages skip Docling.
- **Vault** (§10a) — Notion import (owner step); optional browser Obsidian container; Inbox triage / weekly vault health report / unindexed staging folder discussed and postponed.
- **§17 VERIFY items** still open: `typesafe-sdk` fields + Jev retention, LINE content retention, LiteLLM/Phoenix/Smokescreen tags, least-privilege Copilot token, nodeReact nginx network name.

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
- Qwen3.5 9B Ollama tag and whether it supports vision (resolved 2026-11-04: `qwen3.5:9b` captions images via
  `/api/chat` with `think:false`).
- ctranslate2 CUDA on the RTX 5060 Ti (Blackwell) (resolved 2026-11-04: ctranslate2 4.8.5 float16 works; large-v3-turbo loads in ~1 s).
- LINE reply-token expiry and content retention window (content API reachable through squid, 2026-11-04; retention still unverified, heavy media acks with the reply token at once).
- Copilot CLI `--allow-tool` / `--deny-tool` pattern syntax (resolved 2026-10-28: `copilot help permissions` —
  `shell(cmd)`, `shell(git:*)` prefix form, `write(path?)`, `url(…)`; deny beats allow); ACP permission message format; terms for non-coding use.
- LiteLLM, Phoenix, Smokescreen image tags and config keys.
- Copilot CLI: honours `HTTPS_PROXY` (resolved 2026-10-28: all Copilot traffic seen in the squid log);
  least-privilege token type; usage/credit field names (resolved: `session.usage_checkpoint`).
- Name of the nodeReact nginx Docker network.
- Obsidian Importer: Notion API support and database → Bases mapping (resolved 2026-11-04, §10a); linuxserver/obsidian image (open; optional, not built).
- §12a: Brave `freshness` parameter and per-query price; whether the LINE reply token outlives a clarify round (else use push); local KB for PRIVATE verification (§10a shipped 2026-11-04; wiring `kb_search`/`kb_read` is still open).
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
- Monthly restore test (`host/restore-test.sh`, timer `agent-restore-test.timer`, 1st of the month 05:00): restores `pg/agent.dump` into a throwaway `pgvector/pgvector:pg17` container (`--network none`), counts decisions/todos/chunks, alerts to LINE on pass or fail, removes the container.

### Schedules, morning brief, events, eval (phase 11, 2026-11-04)
- **Schedules** (`app/schedules.py`, table `schedules`): 5-field cron in Asia/Taipei; `schedules_cron` (agent-scheduler, second 5) claims each due row once per minute. Builtins come from policy `schedules.builtin` (inserted when missing; editable/disable-able, not deletable). Owner schedules are kind `prompt` and run their text as a background cron job in `web:sched-<name>`. REST: `GET/POST /api/agent/schedules`, `PATCH/DELETE /schedules/{id}`, `POST /schedules/{id}/run`.
- **Morning brief** (`app/brief.py`, 07:00, no model call): weather (Open-Meteo, places in policy `brief.weather`), today's calendar, overdue/due todos, Inbox counts, yesterday's spend, quiet alerts of the last 24 h. The full text goes to the web thread `web:brief` (one message per day; not private since 2026-11-06, so follow-ups may use cloud models) and an Inbox item `brief` (the previous one is dismissed). LINE gets only weather, counts and spend — never calendar or todo titles.
- **Events** (`app/events.py`): `POST /events/{source}` needs a source in policy `events.sources` and a secret in `EVENT_SECRETS` (JSON); else 404. Bad HMAC → 401; body ≤256 KB JSON; deduped on `X-Event-Id` / `X-GitHub-Delivery` / body hash. Result: an Inbox item `event` (private unless the source says otherwise) and an optional LINE line. Events never start an agent run (untrusted text).
- **Nightly eval** (`app/tools/answer_eval.py`, 02:30, background queue): `config/answer_eval.yaml` questions run through the real pipeline in throwaway `eval:*` threads (no memory written, not listed); pass = finished without a question and contains an expected string. Plus `rag_eval` recall@5/MRR. Stored in `eval_runs`; a drop vs. the previous run beyond policy `eval` thresholds → LINE alert `eval:regression`. Manual: `docker compose exec agent-worker-bg python -m app.tools.answer_eval -v`.
- **CI**: `.github/workflows/tests.yml` runs the unit tests (no services) on every push.

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

### Today home page with exception-only harness status (nodeReact) — planned (decided 2026-11-05, not built)
One owner-only page `/Today` (candidate nodeReact landing page) that answers "what do I need to do today"; the
harness health view is folded into it and **only shows up when something needs attention** — no separate agent
dashboard. nodeReact constraints: React 17, chart.js **2.9** (v2 API), Node 14 toolchain — see `AGENTS.md`.
- **Today (always shown):** weather, today's calendar, todos overdue / due today (by category, §10d), shopping
  list, pending Inbox approvals and questions, items in progress from `/Logs` (§10d). Essentially the morning brief
  as a live page.
- **Harness status (exception-only):** a single green "all OK" line when nothing is wrong; otherwise a card per
  issue, most severe first, each linking to where it is handled (Inbox, Phoenix trace, `/status`):
  - kill switch not `off`; GPU override pinned; unhealthy container or missing heartbeat (vault-indexer,
    gpu-governor); queue backlog or stuck job;
  - spend or Copilot credits ≥ 80% of the daily budget;
  - nightly eval regression; failed schedule or backup; unresolved alerts from the last 24 h;
  - failed verification items, ingest/parse failures, pending curation proposals (§10c).
- **No trend charts by default.** Spend/route/latency/eval history stays in Phoenix and the `eval_runs` tables; a
  collapsible "details" section can be added later if the owner keeps looking things up.
- **API:** `GET /api/agent/today` (owner token, web only) — one aggregate JSON from cheap SQL + Redis reads (no
  model calls, no Docker API per request: container health comes from the alert watcher's last check, cached in
  Redis). Thresholds reuse the alert watcher / policy values so the page and LINE alerts agree. The page polls every
  60 s; no SSE.
- **Privacy:** owner-only web page, so todo/calendar titles are shown (as on `/Todos`); never exposed to LINE or
  unauthenticated routes. Harness status items carry metadata only.
- **Read-only first.** Controls (clear kill switch, unpin GPU) may come later behind a confirm dialog and the same
  owner checks as `host/killswitch.sh` / `host/gpu-state.sh`; every change audited.
- **Acceptance:** loads in < 1 s from a warm API; with a healthy harness only the "all OK" line appears; stopping
  a container, setting the kill switch or a forced eval regression each produce exactly one card that disappears
  when resolved; counts match `/status` and the morning brief.

### Finance dashboard — planned (draft, decided 2026-11-05; details open)
A separate owner-only nodeReact page (not part of `/Today`): net worth, income/spending, investments.
Builds on nodeReact's existing Stock and Bitfinex pages instead of duplicating them.
- **Data sources** (researched 2026-11-05):
  - Investments: SinoPac **Shioaji** API, scheduled read-only pull (`list_positions`, `list_profit_loss`,
    `settlements`, `margin`, `account_balance`; query-only API key, no CA needed; 25 accounting calls / 5 s). It
    gives only the settlement-account balance, not bank transactions. Crypto: existing Bitfinex integration.
  - Banks and credit cards: monthly e-statement emails (encrypted PDFs). Taiwan Open Banking account data is
    TSP-only, so no personal bank API.
  - E-invoices: the platform's monthly email (format to check: attachment or body, line items or totals only).
  - Fallback: manual CSV upload in nodeReact. Browser automation was considered and rejected (captcha,
    Cloudflare, bank ToS, risk) — revisit only if email fails.
- **Email ingest (new):** a dedicated finance mailbox (forwarding rules from the main inbox), read over IMAP by a
  scheduler job; route by sender, decrypt PDFs locally (passwords in `.env`), extract via the media-worker, parse
  into a `transactions` table (amounts parsed by code, never by a model; dedupe per statement/invoice), failures →
  Inbox item.
- **Rules:** numbers are computed by SQL/code; the AI only explains and answers questions. Finance data is
  owner-marked PRIVATE (local models only; revised privacy model, §6); cloud sees aggregates at most. LINE and the
  morning brief get threshold alerts (e.g. budget 80%) without amounts.
- **Open:** e-invoice email format, categorisation (merchant rules vs local model), budgets, which banks/cards,
  whether aggregates may go to cloud, page layout.

### Health checks & LINE alerts
- Compose `healthcheck` on every service; `alert-watcher` job (agent-scheduler, every minute) checks: unhealthy containers, queue age > threshold, jobs stuck > timeout, LiteLLM spend > 80%, backup failure, disk/ZFS pool > 85% or not ONLINE, gpu-governor heartbeat missing.
- Alerts → LINE push to owner, deduplicated (one alert per issue per hour) + logged to Inbox.
- Quiet alerts (policy `alerts.quiet_prefixes`, default `queue:` = a job waiting too long, e.g. during the nightly eval): logged to the `alerts` table only, no LINE/Inbox; the next morning brief lists the last 24 h of them (2026-11-04).

### Planned: notifications on Telegram, LINE replies only (decided 2026-11-06, not built)
LINE push messages count against a monthly quota; reply-token replies are free. Proactive messages move to Telegram
(no monthly quota), and LINE only answers the owner's messages.

**Telegram: notifications only (one-way)**
- One exit `notify(kind, text, link=None)` (e.g. `app/channels/notify.py`) replaces every proactive `line.push`:
  morning brief notice (`brief.py`), todo reminders (`jobs/todo_jobs.py`), event notices (`events.py`), alerts
  (`jobs/alerts.py`), Inbox questions and approvals (`inbox.py`, `tools/calendar.py`, `vault_writer.py`),
  background job done (`graphs/coding.py`, `jobs/worker.py`).
- Telegram Bot API `sendMessage` to the owner's chat id (`TELEGRAM_BOT_TOKEN`, `telegram.owner_chat_id`); add
  `api.telegram.org` to the egress allowlist. Existing rules stay: quiet hours, alert dedup, quiet alerts.
- No commands, chat or buttons on Telegram: approvals and answers link to the web `/Inbox`. Incoming Telegram
  messages are ignored.
- Privacy unchanged: Telegram bot messages are not end-to-end encrypted, so PRIVATE items send only a content-free
  notice (as `events.py` and the calendar/vault approvals do for LINE today).

**LINE: chat only, answered within the reply window**
- LINE already forces the chat route and caps the model at `cloud_fast` (`decision.py` rule 6); the model choice
  stays (local or cloud_fast, as the router picked).
- New: a LINE turn has a deadline below `line_reply_token_ttl_s` (e.g. 35 s of 50 s, VERIFY the real expiry). Near
  the deadline it stops searching/reading and answers with what it has. On LINE a failed L9 check does not
  escalate to a bigger model; the warning is appended instead.
- The push fallback in `line.deliver` is removed: a turn that still misses the window is dropped and logged as a
  quiet alert (no push). Goal: zero LINE push messages per month.
- LINE threads stay chat tier (§12d): follow-up suggestions are chat follow-ups only, no research or deep research.
- Clarify on LINE uses the reply token of the owner's answer (already the case); VERIFY it covers every round.

**Acceptance:** all proactive messages arrive on Telegram, none on LINE; a month of use shows zero LINE pushes; a
LINE answer never arrives after the reply window; PRIVATE notices on Telegram carry no content.

**Open:** what the GAMING (`local_cpu`) path does when it cannot meet the LINE deadline (short "ask on the web"
reply vs. a smaller answer).

### Egress allowlist (`config/egress-allowlist.yaml`)
- Allowed: TypeSafe API, OpenAI, Anthropic, LINE API/content, GitHub (api, github.com, Copilot endpoints — VERIFY list), package registries (npm, PyPI) for sandboxes, Hugging Face (model downloads, can be disabled after setup), Ollama registry.
- Everything else denied and logged; denials appear in Phoenix/alerts.


## Implemented 2026-10-30 (phase 7: Todos + Inbox)

- **Todos:** table `todos` (`app/todos.py`). REST `/api/agent/todos` (GET `?status=open|done|all`, POST, PUT, DELETE); agent tools `list_todos`/`add_todo`/`complete_todo` (assistant route only, no delete); MCP `list_todos`/`add_todo`. Due dates are Asia/Taipei; a bare date means 23:59. Due reminders: see below.
- **Inbox:** table `inbox` (`app/inbox.py`), kinds `approval|question|alert|memory`. REST `/api/agent/inbox` (list+counts, `/count`, `/{id}/read`, `/{id}/dismiss`, bulk `/dismiss` for alert|memory only, `/{id}/answer`).
- **Approvals days later:** coding approvals and background questions create an item (LINE push unless it came from LINE; private threads get a generic text). The LangGraph checkpoint lives in Postgres; if the Redis pending marker expired, `answer` restores it and resumes. It refuses (409) if the thread has moved on. Dismissing an approval = reject (sandbox discarded).
- Alerts and memory writes feed the Inbox; `inbox_cleanup_cron` (04:00) deletes old done/low items (`inbox:` in policy.yaml).
- nodeReact pages `/Todos` and `/Inbox` (approve/reject/answer, dismiss, bulk-dismiss memory/alert). Failed verification after escalation → Inbox `verify` item (phase 10b).

### nodeReact tooling notes (found during phase 7; rules for agents in `AGENTS.md`)

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
- Privacy: `private_keywords` (行事曆/日曆/行程/calendar) force PRIVATE; `tools_for(..., local=True)` is the second guard, so cloud models never get calendar tools. Event text is wrapped in `<untrusted source="calendar">`. (Superseded 2026-11-06 by the revised privacy model, §6: calendar is open to cloud models.)
- Approval design: instead of a graph `interrupt()`, add/delete create an Inbox `approval` item (`detail.calendar`, no run) and the Inbox `answer` route executes it. The existing Inbox UI needed no change; Reject = dismiss. Approving twice is refused (item no longer open).
- Verified live: read of the real calendar, `/local` run proposed an event (Inbox item), Approve created it in Google, then deleted. The delete-proposal path is only covered by tests. The OAuth app is unverified-production (100-user cap, no 7-day token expiry).

- **Calendar vs todo on LINE (2026-10-30):** local Qwen filed every "add ..." as a todo, worse on long LINE threads (it copies earlier tool calls). Fixes, in order of effect: `tools.narrow_for_intent` (deterministic, no model: add verb + time of day and no todo word → `add_todo` removed; explicit todo/reminder → `add_event` removed); the assistant route hides earlier turns' tool calls and keeps `executor.assistant_history_turns` (8; 20 since 2026-11-06) turns (`graphs.main.drop_old_tool_turns`); the prompt rule. A Jev/cloud classifier was rejected: the text is PRIVATE. LINE gets a content-free notice when a proposal was made.
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
| Owner cannot see PRIVATE answers on LINE, so cannot tell what happened (before LINE became never-PRIVATE; now applies to notices from PRIVATE web threads) | The LINE notice is built from this turn's real tool results (added / proposed / failed / nothing happened / the assistant is asking you a question that must be answered on the web). No content and no URLs: it only names the page (Todos, Inbox) | `worker._private_line_notice` |

**Rule for new tools:** a tool that writes anything (todo, calendar, memory, later email) must (1) resolve user-facing
values in code, (2) return a one-line result that starts with a stable prefix (`Added`, `Proposed`, `Not added`) so
those prefixes can drive the checks above, (3) be covered by `narrow_for_intent`/`required_tool` when the request is
recognisable by cheap rules, and (4) never be reported as done unless its tool result says so. Intent rules are
regexes in `tools/__init__.py`; keep them conservative (queries such as "what todos do I have" are excluded) and leave
both tools available when the request is ambiguous.

**Not done / limits:** the rules only cover clear phrasings; an ambiguous message ("新增 買牛奶") can still be filed in
the wrong place. The L9 verify layer (phase 10b) now catches a required write tool that was not called and
tool errors, checking PRIVATE text with local Qwen only; it does not re-run turns that wrote data. Qwen still occasionally writes Simplified Chinese despite the prompt rule.

- **LINE is chat + notify only (2026-10-30):** `decision` rule 6 forces every LINE message to the `chat` route (no todo/calendar/coding tools; models stay local or cloud_fast). A clear add-todo/add-event phrasing on LINE is answered in code ("use the web") before any model runs (`pipeline._line_write_request`), so the model cannot claim it saved something. Todos, calendar and approvals are done on the web. `GET /threads` now returns `private`; the web chat list shows 🔒 and the header shows "Private · local only" for a locked thread (a new chat unlocks). (Since 2026-11-06 only the owner locks a thread, §6.)
- **LINE thread reset (2026-10-30):** the first LINE thread (`line:<uid>`) was locked PRIVATE with private history, so the owner asked for a clean start. `channels.line.thread_id()` now returns `line:<uid>:2`; the old thread is hidden and never sent anywhere. The web alias `line` maps to the new thread. (Superseded the same day: calendar keywords no longer lock a LINE thread, see below.)

**LINE is never PRIVATE.** `gate0(..., channel="line")` always returns no privacy tag (keywords, sources, patterns and thread stickiness are ignored); secrets and PII are still redacted before any cloud call. Web keeps the private tier.
