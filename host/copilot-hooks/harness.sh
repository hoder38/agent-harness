#!/bin/sh
# Forward a Copilot hook payload to the harness (ARCHITECTURE.md §9b). Never blocks or fails Copilot.
# usage: harness.sh agentStop|sessionEnd   (payload JSON on stdin)
event="${1:-}"
token="$(cat "$HOME/.config/harness/token" 2>/dev/null)" || exit 0
payload="$(cat)"
body=$(printf '%s' "$payload" | python3 -c 'import json,sys
d=json.load(sys.stdin); d["event"]=sys.argv[1]; print(json.dumps(d))' "$event" 2>/dev/null) || body="$payload"
printf '%s' "$body" | curl -s -m 3 -X POST http://127.0.0.1:8100/ingest/copilot \
  -H "Authorization: Bearer $token" -H 'Content-Type: application/json' --data-binary @- >/dev/null 2>&1
exit 0
