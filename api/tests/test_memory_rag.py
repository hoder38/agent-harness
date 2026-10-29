"""Phase 10 (§10): chunking, hybrid-retrieval helpers, document privacy, long-term memory, /remember, grounded KB."""
import hashlib
import json

import fakeredis
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from app import config, memory, pipeline, state
from app.jobs import memory_jobs
from app.models import Job
from app.rag import chunking, store
from app.tools.rag_eval import first_rank, score
from test_grounded import PLAN, Web, inp, run


# ---------------------------------------------------------------- chunking
def test_chunks_keep_exact_headings_and_merge_only_intros_into_subsections():
    md = ("# Spec\n\n## 8. Jobs\n\nRedis and arq queues with retries and cancel keys.\n\n"
          "## 9. Copilot\n\nIntro.\n\n### 9a. Coding route\n\n" + "Sandbox text. " * 50 + "\n\n```\n# not a heading\n```\n")
    chunks = chunking.chunk_markdown(md, min_tokens=400, max_tokens=800)
    heads = [c.heading for c in chunks]
    assert "Spec › 8. Jobs" in heads  # a small section stays its own chunk under its own heading
    sub = next(c for c in chunks if c.heading.endswith("9a. Coding route"))
    assert sub.text.startswith("9. Copilot\n\nIntro.") and "# not a heading" in sub.text  # intro carried, fence kept
    assert not any(h.endswith("9. Copilot") for h in heads)


def test_long_sections_split_under_max_with_overlap():
    paras = [f"Paragraph {i} " + "word " * 60 for i in range(20)]
    chunks = chunking.chunk_markdown("## Long\n\n" + "\n\n".join(paras), min_tokens=100, max_tokens=200, overlap=0.5)
    assert len(chunks) > 3 and all(chunking.est_tokens(c.text) <= 200 for c in chunks)
    first_of_next = chunks[1].text.split("\n\n")[0]
    assert first_of_next in chunks[0].text  # trailing paragraph repeated
    huge = chunking.chunk_markdown("x " * 3000, max_tokens=300)
    assert all(chunking.est_tokens(c.text) <= 300 for c in huge)


def test_keyword_tokens_index_cjk_bigrams_and_drop_stop_words():
    assert chunking.kw_tokens("How does the GPU governor work") == ["gpu", "governor", "work"]
    assert chunking.kw_tokens("無糖綠茶") == ["無糖", "糖綠", "綠茶"]
    assert chunking.tsquery_text("the 綠茶?") == "綠茶"
    assert chunking.tsquery_text("the of !!") is None
    meta, body = chunking.split_frontmatter("---\ntitle: Note\nprivacy: private\n---\n# H\ntext")
    assert meta == {"title": "Note", "privacy": "private"} and body.startswith("# H")


# ---------------------------------------------------------------- fusion / selection
def test_rrf_and_select_hits():
    fused = store.rrf([[1, 2, 3], [3, 1]], k=60)
    assert sorted(fused, key=lambda i: -fused[i]) == [1, 3, 2]
    cands = [{"id": i, "path": p, "heading": h, "vec_sim": v} for i, p, h, v in
             [(1, "a.md", "A", 0.9), (2, "a.md", "A", 0.8), (3, "b.md", "B", 0.3), (4, "c.md", "", 0.6)]]
    # reranker on: order by score, floor, one chunk per (path, heading)
    hits = store.select_hits([dict(c) for c in cands], [0.5, 0.9, 0.05, 0.4], k=5, min_rerank=0.1, min_vec=0.45)
    assert [h["id"] for h in hits] == [2, 4]
    # a near-zero rerank score is kept when the embedding still rates it relevant
    hits = store.select_hits([dict(c) for c in cands], [0.0, 0.0, 0.0, 0.0], k=5, min_rerank=0.01, min_vec=0.45)
    assert [h["id"] for h in hits] == [1, 4]
    # reranker down: fused order, vector-similarity floor
    hits = store.select_hits([dict(c) for c in cands], None, k=5, min_rerank=0.1, min_vec=0.45)
    assert [h["id"] for h in hits] == [1, 4] and hits[0]["score"] is None
    assert store.cite_label({"path": "a.md", "heading": "Spec › 11. GPU"}) == "a.md#11. GPU"


def test_document_privacy():
    rules = {"private_doc_dirs": ["private"], "private_keywords": ["salary"]}
    assert store.doc_privacy("x.md", {"privacy": "private"}, "hi", rules)[0] == "PRIVATE"
    assert store.doc_privacy("notes/Private/x.md", {}, "hi", rules)[0] == "PRIVATE"
    assert store.doc_privacy("x.md", {}, "My Salary is", rules)[0] == "PRIVATE"
    assert store.doc_privacy("x.md", {}, "token ghp_" + "a" * 36, rules)[0] == "PRIVATE"
    # contact details and config examples are redacted on non-private searches, not a reason to hide the doc
    assert store.doc_privacy("x.md", {}, "mail me@example.com\napi_key: os.environ/OPENAI_API_KEY", rules)[0] == "NORMAL"


def test_eval_scoring():
    hits = [{"path": "a.md", "heading": "Spec › Setup"}, {"path": "b.md", "heading": "Usage"}]
    assert first_rank(hits, ["b.md#usage"]) == 2 and first_rank(hits, ["a.md"]) == 1
    assert first_rank(hits, ["a.md#Usage"]) is None
    assert score([1, 2, None, 6], 5) == (0.5, (1 + 0.5) / 4)


# ---------------------------------------------------------------- memory
def _embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in chunking.kw_tokens(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
        out.append(v)
    return out


@pytest.fixture
def mem(monkeypatch):
    s = InMemoryStore(index={"dims": 32, "embed": _embed, "fields": ["text"]})
    memory.set_store(s)
    pol = config.policy()
    monkeypatch.setattr(config, "policy", lambda: {**pol, "memory": {**pol["memory"], "enabled": True,
                                                                     "min_score": 0.1, "extract_context_min_score": 0.1}})
    yield s
    memory.set_store(None)


async def test_private_memories_are_recalled_only_on_private_runs(mem):
    await memory.remember("profile", "The owner drinks green tea", private=False)
    await memory.remember("profile", "The owner's green tea budget is secret", private=True)
    normal = await memory.recall("green tea", private=False)
    assert [m["text"] for m in normal] == ["The owner drinks green tea"]
    assert len(await memory.recall("green tea", private=True)) == 2
    assert "may be outdated" in memory.as_prompt(normal) and "green tea" in memory.as_prompt(normal)


async def test_extractor_ops_touch_only_shown_memories_and_keep_history(mem):
    k = await memory.remember("profile", "The owner uses npm", private=False)
    other = await memory.remember("procedures", "Answer in bullet points", private=False)
    reply = {"ops": [{"op": "update", "key": k, "text": "The owner uses pnpm, not npm"},
                     {"op": "delete", "key": "not-shown"},
                     {"op": "add", "namespace": "weird", "text": "The owner lives in Taipei"}]}
    llm = GenericFakeChatModel(messages=iter([AIMessage("```json\n" + json.dumps(reply) + "\n```")]))
    ops = await memory.extract(llm, "local", user_text="I switched to pnpm npm, and I live in Taipei", answer="ok",
                               private=False, source={"run_id": "r1"})
    assert [o["op"] for o in ops] == ["update", "add"]
    m = await memory.get("profile", k)
    assert m["text"] == "The owner uses pnpm, not npm" and m["history"][0]["text"] == "The owner uses npm"
    assert (await memory.get("procedures", other))["text"] == "Answer in bullet points"
    added = next(o for o in ops if o["op"] == "add")
    assert added["namespace"] == "profile" and (await memory.get("profile", added["key"]))["source"] == {"run_id": "r1"}


async def test_remember_command(redis, mem):
    async def enq(*a):
        raise AssertionError("commands are not enqueued")

    def job(text, key, tid="web:t"):
        return Job(channel="web", user_id="owner", thread_id=tid, text=text, dedupe_key=key)

    res = await pipeline.submit(job("/remember I prefer window seats", "a"), enq)
    assert res.command_reply == "🧠 Remembered: I prefer window seats"
    res = await pipeline.submit(job("/remember my email is me@example.com", "b"), enq)
    assert res.command_reply.startswith("🧠 Remembered (private)")
    await state.mark_thread_private(redis, "web:p")
    await pipeline.submit(job("/remember I like jazz", "c", "web:p"), enq)
    assert (await pipeline.submit(job("/remember", "d"), enq)).command_reply.startswith("Usage: /remember")
    items = {m["text"]: m["privacy"] for m in await memory.search(None, None)}
    assert items == {"I prefer window seats": "normal", "my email is me@example.com": "private",
                     "I like jazz": "private"}
    assert "/remember" in pipeline.HELP_TEXT


class Arq(fakeredis.FakeAsyncRedis):
    jobs: list

    async def enqueue_job(self, fn, payload, **kw):
        self.jobs.append((fn, payload, kw))


async def test_after_turn_queues_extraction_and_episode_cron(mem, monkeypatch):
    r = Arq(decode_responses=True)
    r.jobs = []
    await memory_jobs.after_turn(r, run_id="r1", thread_id="web:t", user_text="I moved to Tainan", answer="ok",
                                 private=True, channel="web")
    await memory_jobs.after_turn(r, run_id="r2", thread_id="web:t", user_text="/status", answer="", private=False,
                                 channel="web")
    assert [(f, p["run_id"], p["private"], kw["_queue_name"]) for f, p, kw in r.jobs] == [
        ("extract_memory", "r1", True, "arq:background")]
    r.jobs.clear()
    await memory_jobs.episodes_cron({"redis": r})
    assert not r.jobs  # not idle yet
    monkeypatch.setattr(memory_jobs.time, "time", lambda: 10 ** 10)
    await memory_jobs.episodes_cron({"redis": r})
    await memory_jobs.episodes_cron({"redis": r})
    assert [(f, p) for f, p, _ in r.jobs] == [("summarize_episode", {"thread_id": "web:t"})]
    await r.aclose()


# ---------------------------------------------------------------- grounded graph: KB + memories
KB_HIT = {"path": "agent-harness/ARCHITECTURE.md", "heading": "Spec › 11. GPU governor", "title": "Spec",
          "text": "GAMING unloads Ollama models.", "date": None, "score": 0.9}


class KB:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    async def __call__(self, query, *, private):
        self.calls.append(private)
        return [dict(h) for h in self.hits]


@pytest.fixture
def kbgraph(redis, policy):
    from app.graphs.grounded import build_grounded_graph
    from test_grounded import Clock, TaskLLM
    from langgraph.checkpoint.memory import InMemorySaver

    pol = {**policy, "grounding": {**policy["grounding"], "mode": "cite_only", "cite_pages": 1}}

    def _make(script, web=None, hits=(KB_HIT,), mems=()):
        llm = TaskLLM(script=script, calls=[], seen=[])
        web = web or Web()
        kb = KB(list(hits))

        async def recall(query, *, private):
            return [dict(m) for m in mems]

        g = build_grounded_graph(llm, policy=pol, checkpointer=InMemorySaver(), search=web.search, fetch=web.fetch,
                                 kb=kb, recall=recall, clock=Clock())
        return g, llm, web, kb
    return _make


def _system(llm, task):
    return next(m for t, m in llm.seen if t == task)[0].content


async def test_kb_sources_are_cited_by_path_before_web_sources(kbgraph):
    web = Web(hits={"pricing": ["https://openai.com/p"]}, pages={"https://openai.com/p": "price"})
    g, llm, _, kb = kbgraph({"plan_search": PLAN, "compose": "Notes say [1]; web says [2]."}, web)
    out, _ = await run(g, inp("What happens when gaming starts and what is OpenAI pricing?"))
    text = out["answer"]["text"]
    assert "[1] 📄 Spec › 11. GPU governor — agent-harness/ARCHITECTURE.md#11. GPU governor" in text
    assert "[2] Title https://openai.com/p — https://openai.com/p" in text
    assert "owner's own documents already returned: Spec › 11. GPU governor" in _system(llm, "plan_search")
    human = next(m for t, m in llm.seen if t == "compose")[-1].content
    assert '<untrusted source="[1] kb:agent-harness/ARCHITECTURE.md#11. GPU governor">' in human
    assert kb.calls == [False]


async def test_private_thread_gets_kb_and_memories_but_no_web(kbgraph):
    mem = {"namespace": "profile", "text": "The owner games on Steam", "updated_at": None}
    g, llm, web, kb = kbgraph({"plan_search": PLAN, "compose": "Per your notes [1]."}, mems=[mem])
    out, _ = await run(g, inp("what happens to my GPU jobs?", private=True))
    assert kb.calls == [True] and not web.searches and "plan_search" not in llm.calls
    assert "📄" in out["answer"]["text"] and out["answer"]["verdict"] == "cited"
    sysmsg = _system(llm, "compose")
    assert "Notes about the owner" in sysmsg and "The owner games on Steam" in sysmsg


async def test_kb_answer_skips_web_unless_needed_and_chitchat_skips_kb(kbgraph):
    g, llm, web, _ = kbgraph({"plan_search": {"needs_web": False}, "compose": "From notes [1]."})
    await run(g, inp("Does the governor need 8 GB of VRAM?"))  # a digit: the fact heuristic would force web
    assert not web.searches  # documents answered it, so the heuristic does not force a web search
    g, llm, web, kb = kbgraph({"plan_search": {"needs_web": False}, "compose": "Hi!"})
    out, _ = await run(g, inp("hi there", task_type="chitchat"))
    assert kb.calls == [] and out["answer"]["verdict"] == "no_sources"


async def test_kb_is_searched_once_before_clarifying(kbgraph):
    g, llm, _, kb = kbgraph({"plan_search": {"needs_web": False}, "compose": "From notes [1]."})
    await run(g, inp("what does it do when I start a game?", force_clarify=True))
    # a sure hit answers "it" (even when the router flagged the request ambiguous) and is reused by research
    assert "assess_ambiguity" not in llm.calls and kb.calls == [False]
    weak = {**KB_HIT, "score": 0.1}
    g, llm, _, kb = kbgraph({"assess_ambiguity": {"ambiguous": False}, "plan_search": {"needs_web": False},
                             "compose": "From notes [1]."}, hits=[weak],
                            mems=[{"namespace": "profile", "text": "The owner plays on Steam", "updated_at": None}])
    await run(g, inp("what does it do when I start a game?"))
    sysmsg = _system(llm, "assess_ambiguity")
    assert "owner's own documents have passages" in sysmsg and "- The owner plays on Steam" in sysmsg
    assert kb.calls == [False] and "The owner plays on Steam" in _system(llm, "compose")
