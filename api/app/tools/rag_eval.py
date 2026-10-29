"""Retrieval eval (§10): recall@k and MRR of the KB search over config/rag_eval.yaml.

  docker compose exec agent-worker-bg python -m app.tools.rag_eval [--k 5] [--no-rerank] [--both] [-v]

A case is a hit when one of its `expect` targets is among the top k: "path" matches any chunk of that document,
"path#Heading" also needs the chunk's heading path to contain "Heading" (case-insensitive). Each run is stored in
the rag_eval_runs table so changes to chunking/embedding/reranking can be compared over time.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from .. import db
from ..config import load_yaml
from ..rag import store
from ..settings import get_settings



def matches(hit: dict[str, Any], target: str) -> bool:
    path, _, heading = target.partition("#")
    if hit["path"] != path:
        return False
    return not heading or heading.lower() in (hit.get("heading") or "").lower()


def first_rank(hits: list[dict[str, Any]], expect: list[str]) -> int | None:
    for i, h in enumerate(hits, 1):
        if any(matches(h, t) for t in expect):
            return i
    return None


def score(ranks: list[int | None], k: int) -> tuple[float, float]:
    """(recall@k, MRR@k) over cases; rank None or > k counts as a miss."""
    if not ranks:
        return 0.0, 0.0
    hit = [r for r in ranks if r is not None and r <= k]
    return len(hit) / len(ranks), sum(1 / r for r in hit) / len(ranks)


async def run(cases: list[dict[str, Any]], k: int, use_rerank: bool, verbose: bool) -> dict[str, Any]:
    ranks: list[int | None] = []
    detail = []
    t0 = time.monotonic()
    for c in cases:
        expect = c["expect"] if isinstance(c["expect"], list) else [c["expect"]]
        hits = await store.search(c["q"], private=True, k=k, use_rerank=use_rerank)
        rank = first_rank(hits, expect)
        ranks.append(rank)
        detail.append({"q": c["q"], "expect": expect, "rank": rank, "top": [store.cite_label(h) for h in hits]})
        if verbose or rank is None:
            print(f"{'✓' if rank else '✗'} [{rank or '-'}] {c['q']}\n    expect {expect}\n    got    "
                  + "\n           ".join(store.cite_label(h) + f"  ({h.get('score')})" for h in hits))
    recall, mrr = score(ranks, k)
    ms = int((time.monotonic() - t0) * 1000 / max(len(cases), 1))
    async with db.get_pool().connection() as conn:
        await conn.execute("INSERT INTO rag_eval_runs (k, n, recall, mrr, rerank, detail) VALUES (%s,%s,%s,%s,%s,%s)",
                           (k, len(cases), recall, mrr, use_rerank, Jsonb({"ms_per_query": ms, "cases": detail})))
    return {"k": k, "n": len(cases), "rerank": use_rerank, f"recall@{k}": round(recall, 3), "mrr": round(mrr, 3),
            "ms_per_query": ms}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--both", action="store_true", help="run with and without the reranker")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--file", default=str(Path(get_settings().policy_path).with_name("rag_eval.yaml")))
    a = ap.parse_args()
    cases = load_yaml(a.file).get("cases") or []
    await db.open_db()
    try:
        modes = [True, False] if a.both else [not a.no_rerank]
        for m in modes:
            print(json.dumps(await run(cases, a.k, m, a.verbose), ensure_ascii=False))
    finally:
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
