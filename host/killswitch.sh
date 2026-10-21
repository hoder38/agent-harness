#!/usr/bin/env bash
# Kill switch (§19):  host/killswitch.sh off|cloud|all   (no arg = show)
set -euo pipefail
cd "$(dirname "$0")/.."
pw=$(grep '^REDIS_PASSWORD=' .env | cut -d= -f2-)
rc() { docker compose exec -T agent-redis redis-cli --no-auth-warning -a "$pw" "$@"; }
case "${1:-}" in
  off|cloud|all) rc SET killswitch "$1" >/dev/null; echo "killswitch=$1" ;;
  "") echo "killswitch=$(rc GET killswitch)" ;;
  *) echo "usage: $0 [off|cloud|all]" >&2; exit 2 ;;
esac
