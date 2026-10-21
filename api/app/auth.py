"""Auth: nodeReact-issued HS256 JWT (§4a) and the host-only harness token."""
import hmac

import jwt
from fastapi import Header, HTTPException, Query, status

from .settings import get_settings


def verify_jwt(token: str) -> str:
    s = get_settings()
    try:
        claims = jwt.decode(
            token,
            s.agent_jwt_secret,
            algorithms=["HS256"],
            audience=s.jwt_audience,
            issuer=s.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}") from e
    if claims["sub"] != s.owner_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not the owner")
    return claims["sub"]


async def require_owner(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),  # EventSource cannot set headers; strip ?token= from nginx logs
) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return verify_jwt(authorization[7:].strip())
    if token:
        return verify_jwt(token)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing token")


async def require_harness_token(authorization: str | None = Header(default=None)) -> None:
    expected = get_settings().harness_token
    got = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
    if not expected or not hmac.compare_digest(got, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad harness token")
