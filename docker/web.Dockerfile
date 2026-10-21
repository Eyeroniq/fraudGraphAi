# Static web console served by Caddy, which also proxies /api to the API (D9).
#   docker build -f docker/web.Dockerfile -t fraudgraph-web .

FROM node:22-alpine AS build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web ./
RUN npm run build


FROM caddy:2-alpine
COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /web/out /srv
EXPOSE 8080
