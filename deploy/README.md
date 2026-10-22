# Deployment runbook

FraudGraph runs as one docker compose stack:

- Caddy, which serves the console and proxies `/api` to the API;
- the API;
- the worker;
- Redis, which holds the feature state;
- Postgres, which holds the decision log;
- a one-shot migration service.

There are two ways to run it:

- **Locally**, `docker-compose.yml` builds images from the source.
- **In production**, `docker-compose.prod.yml` adds released images from GHCR, automatic HTTPS
  and restart policies.

## Host requirements

- A Linux VPS with 2 vCPU, 4 GB RAM and 20 GB disk.
- Docker Engine with Compose v2.24 or later.
- A DNS A/AAAA record pointing the domain at the host.
- Ports 80 and 443 open; Caddy needs both to get its certificate.
- Nothing else is public: the API's port 8000 and `/metrics` stay on the internal network.

## First deploy

1. **Clone the repository** to `/srv/fraudgraph` and work from there:

   ```bash
   git clone https://github.com/Eyeroniq/fraudGraphAi.git /srv/fraudgraph
   ```

   ```bash
   cd /srv/fraudgraph
   ```

2. **Make keys.** Create one key per role with `uv run fgx api-key --scope <read|write|admin>`,
   or use the API image:

   ```bash
   docker run --rm ghcr.io/eyeroniq/fraudgraph-api:<tag> fgx api-key --scope read
   ```

   Make keys for these roles:

   | Key | Scope | Purpose |
   |---|---|---|
   | browser | read | public demo mode; visitors can look but not change anything |
   | worker | write | replayer posts |
   | admin | admin | promotion, simulator control; for you only, never in the browser |

3. **Write `deploy/.env`.** It is git-ignored. Fill in these values:

   | Variable | Value |
   |---|---|
   | `POSTGRES_PASSWORD` | a long random string |
   | `FG_API__API_KEYS` | the `scope:hash` entries, comma-separated |
   | `FG_WEB_API_KEY` | the read key (plain) |
   | `FG_WORKER__WRITE_KEY` | the write key |
   | `FG_WORKER__ADMIN_KEY` | the admin key |
   | `FG_VERSION` | the release tag to run, e.g. `v0.9.0` |
   | `FG_DOMAIN` | the domain, e.g. `fraudgraph.example.com` |
   | `FG_ACME_EMAIL` | the address for certificate expiry notices |

4. **Put the data in place.** The model bundle goes in `artifacts/bundles/`, from the
   full-pipeline workflow's `bundle` artifact or from `fgx bundle`. The replayer's events go in
   `data/processed/paysim.parquet`, from `fgx data prepare`. The containers run as uid 10001:

   ```bash
   sudo chown -R 10001:10001 artifacts reports
   ```

5. **Start the stack:**

   ```bash
   docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml up -d
   ```

6. **Check it:**

   ```bash
   docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml ps
   ```

   ```bash
   curl -fsS https://$FG_DOMAIN/api/ready
   ```

## Deploy a new release

Tagging `vX.Y.Z` on `main` builds and pushes the images (`release.yml`).

1. Set `FG_VERSION=vX.Y.Z` in `deploy/.env`.
2. Pull the new images:

   ```bash
   docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml pull
   ```

3. Restart on them:

   ```bash
   docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml up -d
   ```

Migrations run first, through the `migrate` service. Feature state survives in Redis, and a new
bundle is picked up at start, or hot-swapped by promotion.

## Roll back

Set `FG_VERSION` back to the previous tag, then pull and restart the same way. Alembic
migrations are additive; if a release added one, it is safe to run the older image against
the newer schema. To undo a promoted model without a redeploy, promote the previous version:

```bash
curl -X POST -H "X-API-Key: $ADMIN" https://$FG_DOMAIN/api/v1/models/<version>/promote
```

## Rotate keys

1. Make the new key with `fgx api-key`.
2. Add its entry to `FG_API__API_KEYS` alongside the old one, then restart the stack. Both
   keys work at this point.
3. Switch the client (browser, worker or you) to the new key.
4. Remove the old entry and restart again.

## Backups

`deploy/backup.sh` writes a compressed `pg_dump` and keeps the newest 14. Run it nightly from
cron:

```
0 3 * * * /srv/fraudgraph/deploy/backup.sh /var/backups/fraudgraph
```

To restore a dump:

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml exec -T postgres \
  pg_restore -U fraudgraph -d fraudgraph --clean --if-exists < /var/backups/fraudgraph/<file>.dump
```

Redis holds only feature state, which can be rebuilt by replaying the stream, so it is not
backed up. Copy the bundles and reports along with the database.

## Operating notes

- **Public demo mode.** Caddy injects the read key into every `/api` request, so visitors can
  browse but not score, label, promote or control the replayer. Rate limits apply per key.
- **Metrics.** Scrape `api:8000/metrics` from inside the compose network; Caddy returns 404 for
  `/api/metrics`.
- **Live stream.** With more than one API worker, `/ws/live` shows the decisions made by the
  worker the browser happens to be connected to. Fanning out through Redis pub/sub would fix
  that.
