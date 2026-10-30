#!/usr/bin/env python3
"""One-time Google Calendar consent. Stdlib only; run on the host (not in a container).

  python3 host/google_auth.py [--env /home/hoder/nodeReact/.env]

Reads GOOGLE_ID / GOOGLE_SECRET from the given env file (values are never printed), prints a consent URL, and asks
you to paste the URL your browser lands on afterwards (it shows a connection error on localhost: that is expected).
Writes ~/.config/harness/google/calendar.json (0600), which the agent containers mount read-only. This is a separate
grant from nodeReact's Drive token, so the two never invalidate each other.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SCOPE = "https://www.googleapis.com/auth/calendar.events"
REDIRECT = "http://localhost:8765/"
OUT = Path.home() / ".config/harness/google/calendar.json"


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(GOOGLE_ID|GOOGLE_SECRET)\s*=\s*(.*?)\s*$", line)
        if m:
            env[m.group(1)] = m.group(2).strip("'\"")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="/home/hoder/nodeReact/.env")
    args = ap.parse_args()
    env = read_env(Path(args.env))
    if not env.get("GOOGLE_ID") or not env.get("GOOGLE_SECRET"):
        print("GOOGLE_ID / GOOGLE_SECRET not found in", args.env)
        return 1
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": env["GOOGLE_ID"], "redirect_uri": REDIRECT, "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})
    print("1. Open this URL in your browser and approve:\n\n" + url + "\n")
    print("2. The browser will end on a page that fails to load (http://localhost:8765/?code=...).")
    print("   Copy the full address from the address bar and paste it here.\n")
    landed = urllib.parse.urlparse(input("Pasted URL: ").strip())
    q = urllib.parse.parse_qs(landed.query)
    if q.get("state", [""])[0] != state or "code" not in q:
        print("Missing code or state mismatch:", q.get("error", ["?"])[0])
        return 1
    data = urllib.parse.urlencode({
        "code": q["code"][0], "client_id": env["GOOGLE_ID"], "client_secret": env["GOOGLE_SECRET"],
        "redirect_uri": REDIRECT, "grant_type": "authorization_code", "code_verifier": verifier}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request("https://oauth2.googleapis.com/token", data=data)) as r:
            tok = json.load(r)
    except urllib.error.HTTPError as e:
        print("Token exchange failed:", e.read().decode()[:300])
        return 1
    if "refresh_token" not in tok:
        print("No refresh_token returned; revoke the app at myaccount.google.com/permissions and retry.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"client_id": env["GOOGLE_ID"], "client_secret": env["GOOGLE_SECRET"],
                               "refresh_token": tok["refresh_token"], "scope": tok.get("scope", SCOPE)}))
    os.chmod(OUT, 0o600)
    print("Saved", OUT, "scope:", tok.get("scope"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
