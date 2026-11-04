#!/usr/bin/env bash
# Monthly restore test (§19, phase 11): restore the latest backup dump into a throwaway Postgres, run smoke queries,
# report to LINE either way, clean up. Run as hoder (docker group): host/restore-test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DUMP=${DUMP:-/tank/agent-backup/pg/agent.dump}
NAME=agent-restore-test-$$
TOKEN_FILE=~/.config/harness/token

alert() {
  curl -s -m 5 -X POST http://127.0.0.1:8100/internal/alert \
    -H "Authorization: Bearer $(cat "$TOKEN_FILE" 2>/dev/null)" -H 'Content-Type: application/json' \
    -d "{\"key\":\"$1\",\"message\":\"$2\"}" >/dev/null || true
}
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'alert "restore-test:failed" "agent-harness restore test FAILED at line $LINENO (dump: $DUMP)"' ERR

[[ -s $DUMP ]] || { alert "restore-test:failed" "restore test: no dump at $DUMP"; exit 1; }
age_h=$(( ($(date +%s) - $(stat -c %Y "$DUMP")) / 3600 ))

image=$(docker compose config --images agent-postgres 2>/dev/null | head -1)
docker run -d --name "$NAME" --network none -e POSTGRES_PASSWORD=restore -e POSTGRES_USER=agent \
  -e POSTGRES_DB=agent "${image:-pgvector/pgvector:pg17}" >/dev/null
for _ in $(seq 60); do
  docker exec "$NAME" pg_isready -U agent -d agent -q 2>/dev/null && break
  sleep 1
done
sleep 2  # the image restarts postgres once after init

t0=$(date +%s)
docker exec -i "$NAME" pg_restore -U agent -d agent --no-owner --exit-on-error < "$DUMP"
q() { docker exec "$NAME" psql -U agent -d agent -Atc "$1"; }
decisions=$(q "SELECT count(*) FROM decisions")
todos=$(q "SELECT count(*) FROM todos")
chunks=$(q "SELECT count(*) FROM chunks")
last=$(q "SELECT to_char(max(created_at) AT TIME ZONE 'Asia/Taipei', 'MM-DD HH24:MI') FROM decisions")
(( decisions > 0 ))

msg="✅ restore test OK in $(( $(date +%s) - t0 ))s: decisions=$decisions (last $last), todos=$todos, chunks=$chunks; dump ${age_h}h old"
echo "$msg"
alert "restore-test:ok" "$msg"
