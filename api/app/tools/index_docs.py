"""Index the docs folder now:  docker compose exec agent-worker-bg python -m app.tools.index_docs"""
import asyncio
import json

from .. import db
from ..rag import indexer


async def main() -> None:
    await db.open_db()
    try:
        print(json.dumps(await indexer.index_docs(), ensure_ascii=False, indent=2))
    finally:
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
