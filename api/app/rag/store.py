"""RAG tables, indexing and hybrid retrieval (§10).

documents/chunks live in Postgres: pgvector HNSW (cosine) + a 'simple' tsvector of `chunking.kw_tokens`
(Latin words + CJK bigrams). Query: vector top-N and keyword top-N → reciprocal rank fusion → CPU cross-encoder
rerank of the best `rerank_candidates` → top-k above a score floor, each with a `path#heading` citation.

Privacy: every document gets a privacy_tag (frontmatter, folder rules, keywords, secret/PII patterns). PRIVATE
chunks are returned only to PRIVATE runs, which are pinned to local models (§5).
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from .. import config, db
from ..gates.gate0 import BUILTIN_PATTERNS, _compile_extra, redact
from . import chunking
from .embed import embed, rag_cfg, rerank

log = logging.getLogger(__name__)

DIMS = 1024  # qwen3-embedding:0.6b; changing the model needs a re-index (DROP TABLE chunks)

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,            -- docs | upload (| vault in 10a)
    path        TEXT NOT NULL,
    title       TEXT,
    hash        TEXT NOT NULL,            -- raw text + privacy + chunker params: re-index when it changes
    privacy_tag TEXT NOT NULL,
    privacy_why TEXT,
    content     TEXT,                     -- uploads only (docs are read from the folder)
    doc_date    TIMESTAMPTZ,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_chunks    INT NOT NULL DEFAULT 0,
    UNIQUE (source, path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         INT NOT NULL,
    heading     TEXT,
    text        TEXT NOT NULL,
    embedding   vector({DIMS}),
    tsv         tsvector,
    source      TEXT NOT NULL,
    path        TEXT NOT NULL,
    date        TIMESTAMPTZ,
    privacy_tag TEXT NOT NULL,
    hash        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS rag_eval_runs (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    k           INT NOT NULL,
    n           INT NOT NULL,
    recall      REAL NOT NULL,
    mrr         REAL NOT NULL,
    rerank      BOOLEAN NOT NULL,
    detail      JSONB
);
"""


def vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ---------------------------------------------------------------- privacy
# `api_key: os.environ/X` in a config example is not a secret; such matches are redacted, not a reason to hide a doc.
_DOC_HEURISTIC = {"assigned_secret"}


def doc_privacy(path: str, meta: dict[str, Any], text: str, rules: dict[str, Any]) -> tuple[str, str]:
    """(PRIVATE|NORMAL, reason). Whole document: one private paragraph makes every chunk private."""
    fm = str(meta.get("privacy") or "").strip().lower()
    if fm == "private":
        return "PRIVATE", "frontmatter privacy: private"
    parts = [p.lower() for p in path.split("/")[:-1]]
    for d in rules.get("private_doc_dirs") or []:
        if d.strip("/").lower() in parts:
            return "PRIVATE", f"folder {d}/"
    low = text.lower()
    for kw in rules.get("private_keywords") or []:
        if kw.lower() in low:
            return "PRIVATE", f"keyword '{kw}'"
    for pat in BUILTIN_PATTERNS + _compile_extra(rules):
        if (pat.kind != "secret" and not pat.private) or pat.name in _DOC_HEURISTIC:
            continue  # plain PII / `key: value` heuristics are redacted on non-private searches instead
        for m in pat.regex.finditer(text):
            if not pat.validator or pat.validator(m.group(0)):
                return "PRIVATE", f"contains {pat.kind} ({pat.name})"
    return "NORMAL", "frontmatter privacy: normal" if fm == "normal" else ""


# ---------------------------------------------------------------- indexing
def _date(meta: dict[str, Any], fallback: datetime | None) -> datetime | None:
    v = meta.get("date") or meta.get("updated") or meta.get("created")
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if v is not None:
        try:
            d = datetime.fromisoformat(str(v))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback


async def upsert_document(*, source: str, path: str, raw: str, rules: dict[str, Any],
                          mtime: datetime | None = None, store_content: bool = False,
                          title: str | None = None, force_private: bool = False) -> dict[str, Any]:
    """Index one document; unchanged documents (same hash) are skipped. Unchanged chunks keep their embedding."""
    cfg = rag_cfg()
    meta, body = chunking.split_frontmatter(raw)
    privacy, why = ("PRIVATE", "marked private") if force_private else doc_privacy(path, meta, raw, rules)
    params = (chunking.VERSION, cfg.get("chunk_min_tokens", 400), cfg.get("chunk_max_tokens", 800),
              cfg.get("chunk_overlap", 0.15), cfg.get("embed_model"))
    h = chunking.sha256(raw + "\x00" + privacy + "\x00" + repr(params))  # new chunker/params/model → re-index
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT id, hash FROM documents WHERE source=%s AND path=%s", (source, path))
        row = await cur.fetchone()
        if row and row["hash"] == h:
            return {"path": path, "status": "unchanged", "id": row["id"]}
        old: dict[str, str] = {}
        if row:
            cur = await conn.execute("SELECT hash, embedding::text AS e FROM chunks WHERE doc_id=%s", (row["id"],))
            old = {r["hash"]: r["e"] for r in await cur.fetchall()}
    title = title or chunking.doc_title(meta, body, path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    chunks = chunking.chunk_markdown(body, min_tokens=int(cfg.get("chunk_min_tokens", 400)),
                                     max_tokens=int(cfg.get("chunk_max_tokens", 800)),
                                     overlap=float(cfg.get("chunk_overlap", 0.15)))
    model = str(cfg.get("embed_model"))
    ckey = {c.hash: chunking.sha256(f"{model}\n{title}\n{c.hash}") for c in chunks}  # embedding identity
    todo = [c for c in chunks if ckey[c.hash] not in old]
    vecs = await embed([f"{title}\n{c.heading}\n{c.text}" for c in todo])
    new = {ckey[c.hash]: vec(v) for c, v in zip(todo, vecs)}
    date = _date(meta, mtime)
    async with db.get_pool().connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                """INSERT INTO documents (source, path, title, hash, privacy_tag, privacy_why, content, doc_date,
                                          indexed_at, n_chunks)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now(),%s)
                   ON CONFLICT (source, path) DO UPDATE SET title=EXCLUDED.title, hash=EXCLUDED.hash,
                     privacy_tag=EXCLUDED.privacy_tag, privacy_why=EXCLUDED.privacy_why, content=EXCLUDED.content,
                     doc_date=EXCLUDED.doc_date, indexed_at=now(), n_chunks=EXCLUDED.n_chunks
                   RETURNING id""",
                (source, path, title, h, privacy, why, raw if store_content else None, date, len(chunks)))
            doc_id = (await cur.fetchone())["id"]
            await conn.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))
            for i, c in enumerate(chunks):
                await conn.execute(
                    """INSERT INTO chunks (doc_id, ord, heading, text, embedding, tsv, source, path, date,
                                           privacy_tag, hash)
                       VALUES (%s,%s,%s,%s,%s::vector,to_tsvector('simple', %s),%s,%s,%s,%s,%s)""",
                    (doc_id, i, c.heading, c.text, old.get(ckey[c.hash]) or new[ckey[c.hash]],
                     " ".join(chunking.kw_tokens(f"{title}\n{c.heading}\n{c.text}")),
                     source, path, date, privacy, ckey[c.hash]))
    return {"path": path, "status": "updated" if row else "added", "id": doc_id, "chunks": len(chunks),
            "embedded": len(todo), "privacy": privacy}


async def delete_documents(source: str, keep_paths: set[str] | None = None, *, doc_id: int | None = None) -> int:
    async with db.get_pool().connection() as conn:
        if doc_id is not None:
            cur = await conn.execute("DELETE FROM documents WHERE id=%s AND source=%s", (doc_id, source))
        else:
            cur = await conn.execute("DELETE FROM documents WHERE source=%s AND NOT (path = ANY(%s))",
                                     (source, list(keep_paths or [])))
        return cur.rowcount


async def list_documents(limit: int = 500) -> list[dict[str, Any]]:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            """SELECT id, source, path, title, privacy_tag, privacy_why, doc_date, indexed_at, n_chunks
               FROM documents ORDER BY source, path LIMIT %s""", (limit,))
        return await cur.fetchall()


# ---------------------------------------------------------------- retrieval
def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: Σ 1/(k + rank) over the lists an id appears in (rank from 1)."""
    out: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + rank)
    return out


def select_hits(cands: list[dict[str, Any]], scores: list[float] | None, *, k: int, min_rerank: float,
                min_vec: float) -> list[dict[str, Any]]:
    """Order and cut the fused candidates: by rerank score when available, else by RRF with a vector floor.
    With the reranker a hit is kept if it clears either floor: the cross-encoder gives some relevant chunks
    (paraphrased, cross-language) scores near 0 that the embedding still rates well, while off-topic queries fail both.
    At most one chunk per (path, heading), so overlapping chunks of one section do not crowd out others."""
    if scores is not None:
        for c, s in zip(cands, scores):
            c["score"] = round(s, 4)
        ranked = sorted((c for c in cands if c["score"] >= min_rerank or (c.get("vec_sim") or 0) >= min_vec),
                        key=lambda c: -c["score"])
    else:
        for c in cands:
            c["score"] = None
        ranked = [c for c in cands if (c.get("vec_sim") or 0) >= min_vec]
    seen: set[tuple[str, str]] = set()
    out = []
    for c in ranked:
        key = (c["path"], c.get("heading") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= k:
            break
    return out


async def search(query: str, *, private: bool, k: int | None = None, use_rerank: bool = True,
                 source: str | None = None) -> list[dict[str, Any]]:
    """Hybrid search. Returns [{id, path, title, heading, text, date, privacy_tag, source, score, vec_sim, rrf}]."""
    cfg = rag_cfg()
    query = (query or "").strip()[:2000]
    if not query:
        return []
    k = int(k or cfg.get("top_k", 5))
    n = int(cfg.get("candidates", 50))
    qv = vec((await embed([query], query=True))[0])
    tsq = chunking.tsquery_text(query)
    where = "(%(private)s OR c.privacy_tag <> 'PRIVATE')" + (" AND c.source = %(source)s" if source else "")
    args = {"qv": qv, "tsq": tsq, "n": n, "private": private, "source": source}
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            f"""SELECT c.id, 1 - (c.embedding <=> %(qv)s::vector) AS sim FROM chunks c WHERE {where}
                ORDER BY c.embedding <=> %(qv)s::vector LIMIT %(n)s""", args)
        vec_rows = await cur.fetchall()
        kw_rows = []
        if tsq:
            cur = await conn.execute(
                f"""SELECT c.id FROM chunks c, to_tsquery('simple', %(tsq)s) q WHERE c.tsv @@ q AND {where}
                    ORDER BY ts_rank_cd(c.tsv, q, 1) DESC LIMIT %(n)s""", args)
            kw_rows = await cur.fetchall()
        fused = rrf([[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]], int(cfg.get("rrf_k", 60)))
        top = sorted(fused, key=lambda i: -fused[i])[: int(cfg.get("rerank_candidates", 20))]
        if not top:
            return []
        cur = await conn.execute(
            """SELECT c.id, c.path, c.heading, c.text, c.date, c.privacy_tag, c.source, d.title
               FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ANY(%s)""", (top,))
        rows = {r["id"]: r for r in await cur.fetchall()}
    sims = {r["id"]: float(r["sim"]) for r in vec_rows}
    cands = [{**rows[i], "rrf": round(fused[i], 5), "vec_sim": round(sims.get(i, 0.0), 4)} for i in top if i in rows]
    scores = await rerank(query, [f"{c['title']} › {c['heading']}\n{c['text']}"[:3000] for c in cands]) \
        if use_rerank and cfg.get("rerank", True) else None
    hits = select_hits(cands, scores, k=k, min_rerank=float(cfg.get("min_rerank_score", 0.2)),
                       min_vec=float(cfg.get("min_vector_sim", 0.45)))
    if not private:  # NORMAL docs may still hold an email or phone number: redact before any cloud model sees it
        rules = config.privacy_rules()
        for h in hits:
            h["text"] = redact(h["text"], rules)[0]
    return hits


def cite_label(hit: dict[str, Any]) -> str:
    """`path#heading` (Obsidian-style anchor) for citations."""
    return hit["path"] + (f"#{hit['heading'].split(' › ')[-1]}" if hit.get("heading") else "")


def to_json(hits: list[dict[str, Any]]) -> str:
    return json.dumps([{"cite": cite_label(h), "title": h.get("title"), "score": h.get("score"),
                        "text": re.sub(r"\s+\n", "\n", h["text"])[:1500]} for h in hits], ensure_ascii=False,
                      default=str)
