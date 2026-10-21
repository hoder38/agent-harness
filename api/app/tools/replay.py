"""Decision replay (§19): re-run decide() on stored inputs with a candidate policy — no model calls.

    python -m app.tools.replay --policy /config/policy.next.yaml --since 14d [--limit 5000] [--show 30]
"""
import argparse
import asyncio
import re
from collections import Counter
from datetime import timedelta

import yaml

from .. import db
from ..decision import decide
from ..models import Budget, Decision, Gate0Result, Job, RouterResult


def _since(s: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([dh])", s)
    if not m:
        raise SystemExit("--since must look like 14d or 12h")
    n = int(m.group(1))
    return timedelta(days=n) if m.group(2) == "d" else timedelta(hours=n)


def replay_row(row: dict, pol: dict) -> tuple[Decision, Decision]:
    inputs = row["inputs"]
    job = Job(**{**inputs["job"], "text": row["redacted_text"]})
    g0 = Gate0Result(**{**inputs["gate0"], "text": row["redacted_text"]})
    router = RouterResult(**row["router"])
    old = Decision(**row["decision"])
    new = decide(job, g0, router, inputs["gpu_state"], Budget(**inputs["budget"]), pol,
                 killswitch=inputs.get("killswitch", "off"))
    return old, new


def _cost_estimate(d: Decision, pol: dict, row: dict) -> float:
    usage = ((row.get("outcome") or {}).get("usage")) or {}
    price = (pol.get("prices_usd_per_mtok") or {}).get(d.model, {})
    return (usage.get("input", 0) * price.get("input", 0) + usage.get("output", 0) * price.get("output", 0)) / 1e6


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--since", default="14d")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()
    with open(a.policy, encoding="utf-8") as f:
        pol = yaml.safe_load(f)

    pool = await db.open_db()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM decisions WHERE created_at > now() - %s ORDER BY created_at DESC LIMIT %s",
            (_since(a.since), a.limit),
        )
        rows = await cur.fetchall()
    await db.close_db()

    changes: Counter[str] = Counter()
    old_cost = new_cost = 0.0
    failed_changed = []
    shown = 0
    for row in rows:
        try:
            old, new = replay_row(row, pol)
        except Exception as e:
            print(f"skip {row['id']}: {e}")
            continue
        old_cost += _cost_estimate(old, pol, row)
        new_cost += _cost_estimate(new, pol, row)
        if (old.route, old.model) != (new.route, new.model):
            changes[f"{old.route}/{old.model} → {new.route}/{new.model}"] += 1
            status = (row.get("outcome") or {}).get("status")
            if status in ("error",) or (row.get("outcome") or {}).get("verify") == "fail":
                failed_changed.append(row["id"])
            if shown < a.show:
                shown += 1
                print(f"#{row['id']} {old.route}/{old.model} → {new.route}/{new.model}: {row['redacted_text'][:80]!r}")
                print("   new reasons:", "; ".join(new.reasons))

    print(f"\n{len(rows)} decisions replayed with policy {pol.get('version')}")
    for k, v in changes.most_common():
        print(f"  {v:5d}  {k}")
    print(f"estimated cost (token-based): {old_cost:.4f} → {new_cost:.4f} USD")
    if failed_changed:
        print(f"changed decisions whose old run failed: {failed_changed[:50]}")


if __name__ == "__main__":
    asyncio.run(main())
