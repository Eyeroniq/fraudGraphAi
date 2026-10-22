#!/usr/bin/env sh
# Nightly Postgres backup for the compose stack (task 9.4).
#   deploy/backup.sh [backup-dir]        e.g. from cron: 0 3 * * * /srv/fraudgraph/deploy/backup.sh
# Writes a compressed custom-format dump and keeps the newest KEEP (default 14).
# Restore: see deploy/README.md.
set -eu

cd "$(dirname "$0")"
dir="${1:-/var/backups/fraudgraph}"
keep="${KEEP:-14}"
mkdir -p "$dir"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$dir/fraudgraph-$stamp.dump"

docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
  pg_dump -U fraudgraph -d fraudgraph --format=custom --compress=6 > "$out.tmp"
mv "$out.tmp" "$out"
echo "backup written: $out ($(du -h "$out" | cut -f1))"

# Keep the newest $keep dumps.
ls -1t "$dir"/fraudgraph-*.dump 2>/dev/null | tail -n +"$((keep + 1))" | while read -r old; do
  rm -f -- "$old"
done
