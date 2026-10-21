"""Mint a short-lived owner JWT for manual testing (same shape nodeReact issues, §4a).

    docker compose exec agent-api python -m app.tools.mint_token [--minutes 10]
"""
import argparse
import time
import uuid

import jwt

from ..settings import get_settings


def mint(minutes: int = 10) -> str:
    s = get_settings()
    now = int(time.time())
    return jwt.encode(
        {"sub": s.owner_id, "aud": s.jwt_audience, "iss": s.jwt_issuer, "iat": now,
         "exp": now + minutes * 60, "jti": uuid.uuid4().hex},
        s.agent_jwt_secret, algorithm="HS256",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=10)
    print(mint(ap.parse_args().minutes))
