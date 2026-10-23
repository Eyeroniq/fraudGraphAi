#!/usr/bin/env bash
# Bring up the full compose stack on the synthetic fixture (CI docker-smoke and e2e jobs).
#   deploy/ci-stack.sh          build, start, wait until healthy
# Writes deploy/.env with throwaway keys and exports them to $GITHUB_ENV when present.
set -euo pipefail
cd "$(dirname "$0")/.."

key() { uv run fgx api-key --scope "$1"; }
pick() { sed -n "s/^$1: *//p"; }

write_out=$(key write)
admin_out=$(key admin)
WRITE_KEY=$(echo "$write_out" | pick key)
ADMIN_KEY=$(echo "$admin_out" | pick key)
ENTRIES="$(echo "$write_out" | pick entry),$(echo "$admin_out" | pick entry)"

# The browser's key: admin in CI so the e2e test can drive the simulator and resolve
# cases; a public deployment injects a read-only key instead (deploy/README.md).
cat > deploy/.env <<EOF
POSTGRES_PASSWORD=ci-$(date +%s)
FG_API__API_KEYS=$ENTRIES
FG_WEB_API_KEY=$ADMIN_KEY
FG_WORKER__WRITE_KEY=$WRITE_KEY
FG_WORKER__ADMIN_KEY=$ADMIN_KEY
FG_API_WORKERS=2
FG_API__RATE_LIMIT_SCORE=1000000/minute
FG_ARTIFACTS=artifacts-ci
FG_EVENTS=fixture/events.parquet
FG_WEB_PORT=${FG_WEB_PORT:-8080}
EOF
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "::add-mask::$WRITE_KEY"
  echo "::add-mask::$ADMIN_KEY"
  { echo "STACK_WRITE_KEY=$WRITE_KEY"; echo "STACK_ADMIN_KEY=$ADMIN_KEY"; } >> "$GITHUB_ENV"
fi

# A bundle and replayable events from the synthetic fixture (CI never needs PaySim, D10).
# They go to their own folders, never over real data in artifacts/ or data/processed/.
uv run fgx all --fixture --out artifacts-ci/bundles
mkdir -p data/fixture reports
uv run python -c "from fraudgraph.data.prepare import to_canonical; from fraudgraph.data.synth import generate; to_canonical(generate().frame).write_parquet('data/fixture/events.parquet')"
chmod -R a+rwX artifacts-ci reports  # the containers run as uid 10001

docker compose -f deploy/docker-compose.yml up -d --build
# `up --wait` trips over the one-shot migrate container; poll readiness through Caddy.
for _ in $(seq 1 100); do
  if curl -fsS "http://localhost:${FG_WEB_PORT:-8080}/api/ready" >/dev/null 2>&1; then
    docker compose -f deploy/docker-compose.yml ps
    exit 0
  fi
  sleep 3
done
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs --tail 80
echo "stack did not become ready" >&2
exit 1
