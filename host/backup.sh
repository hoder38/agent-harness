#!/usr/bin/env bash
# Nightly backup to ZFS (§19). Run as hoder (docker group) with ZFS delegation:
#   sudo zfs create -o compression=zstd -o mountpoint=/tank/agent-backup tank/agent-backup
#   sudo chown hoder:hoder /tank/agent-backup
#   sudo zfs allow hoder snapshot,destroy,mount tank/agent-backup
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-tank/agent-backup}
DEST=${DEST:-/tank/agent-backup}
POOL=${DATASET%%/*}
D=$(date +%Y%m%d)
TOKEN_FILE=~/.config/harness/token

alert() {
  curl -s -m 5 -X POST http://127.0.0.1:8100/internal/alert \
    -H "Authorization: Bearer $(cat "$TOKEN_FILE" 2>/dev/null)" -H 'Content-Type: application/json' \
    -d "{\"key\":\"$1\",\"message\":\"$2\"}" >/dev/null || true
}
trap 'alert "backup:failed" "agent-harness backup failed at line $LINENO"' ERR

mkdir -p "$DEST"/{pg,config,vault,volumes}

# 1. Postgres (only the latest dump is kept on disk; history lives in snapshots)
docker compose exec -T agent-postgres pg_dump -U agent -Fc agent > "$DEST/pg/agent.dump.tmp"
mv "$DEST/pg/agent.dump.tmp" "$DEST/pg/agent.dump"

# 2. Files: config (no .env), vault (+ .git) if present, Redis AOF + Phoenix volumes
rsync -a --delete --exclude '.env*' ./config/ "$DEST/config/"
[[ -d /home/hoder/vault ]] && rsync -a --delete /home/hoder/vault/ "$DEST/vault/"
[[ -d /srv/agent/files ]] && mkdir -p "$DEST/files" && rsync -a --delete /srv/agent/files/ "$DEST/files/"
for v in redis phoenix; do
  docker run --rm -v "agent-harness_${v}:/src:ro" -v "$DEST/volumes:/dst" busybox \
    tar -C /src -cf "/dst/${v}.tar" .
done
# .env is backed up only encrypted, if `age` and a recipient are configured
if command -v age >/dev/null && [[ -n "${AGE_RECIPIENT:-}" ]]; then
  age -r "$AGE_RECIPIENT" -o "$DEST/config/env.age" .env
fi

# 3. Snapshot
zfs snapshot "${DATASET}@${D}" 2>/dev/null || zfs list -t snapshot "${DATASET}@${D}" >/dev/null

# 4. Retention: 14 daily, 8 weekly (Sundays), 6 monthly (1st of month)
now=$(date +%s)
zfs list -H -t snapshot -o name -s creation "$DATASET" | while read -r snap; do
  tag=${snap#*@}
  [[ $tag =~ ^[0-9]{8}$ ]] || continue
  ts=$(date -d "$tag" +%s) || continue
  age_d=$(( (now - ts) / 86400 ))
  keep=0
  (( age_d < 14 )) && keep=1
  [[ $(date -d "$tag" +%u) == 7 ]] && (( age_d < 56 )) && keep=1
  [[ $(date -d "$tag" +%d) == 01 ]] && (( age_d < 186 )) && keep=1
  (( keep )) || zfs destroy "$snap"
done

# 5. Pool health
health=$(zpool list -H -o health "$POOL")
cap=$(zpool list -H -o capacity "$POOL" | tr -d '%')
[[ $health == ONLINE ]] || alert "zfs:health" "ZFS pool $POOL is $health"
(( cap < 85 )) || alert "zfs:capacity" "ZFS pool $POOL is ${cap}% full"
echo "backup ok: ${DATASET}@${D}"
