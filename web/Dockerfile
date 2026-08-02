# ---------------------------------------------------------------------------------
# web/Dockerfile — builds the React/Vite SPA and serves it with nginx.
#
# Build context is `web/` (docker-compose.yml sets `context: ./web`):
#
#   docker build -t rag-web:dev --build-arg VITE_API_BASE_URL=http://localhost:8000 ./web
#
# Vite inlines VITE_* values at build time, so they are build arguments, not runtime
# environment variables. The only runtime knobs are the nginx ones (API_UPSTREAM,
# NGINX_PORT), substituted into nginx.conf by the nginx image's envsubst entrypoint.
#
# Leave VITE_API_BASE_URL empty to make the SPA call the API same-origin through the
# /api reverse proxy in nginx.conf — that is the recommended production setup, because
# it needs no CORS configuration and no rebuild when the API hostname changes.
# ---------------------------------------------------------------------------------

# syntax=docker/dockerfile:1.7

ARG NODE_VERSION=25
ARG NGINX_VERSION=1.27

# --------------------------------------------------------------------------- build
FROM node:${NODE_VERSION}-alpine AS build

ENV CI=true \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

WORKDIR /app

# The whole context is copied before installing on purpose: `npm ci` recreates
# node_modules from scratch, so a stale host node_modules built for another platform
# (darwin-arm64 esbuild, for example) can never leak into the image.
COPY . .

RUN if [ -f package-lock.json ]; then \
        npm ci; \
    else \
        echo "no package-lock.json: falling back to npm install" >&2; \
        npm install; \
    fi

ARG VITE_API_BASE_URL=""
ARG VITE_ENTRA_CLIENT_ID=""
ARG VITE_ENTRA_TENANT_ID=""
ARG VITE_ENTRA_API_SCOPE=""
ARG VITE_DEV_MODE="false"

ENV VITE_API_BASE_URL=${VITE_API_BASE_URL} \
    VITE_ENTRA_CLIENT_ID=${VITE_ENTRA_CLIENT_ID} \
    VITE_ENTRA_TENANT_ID=${VITE_ENTRA_TENANT_ID} \
    VITE_ENTRA_API_SCOPE=${VITE_ENTRA_API_SCOPE} \
    VITE_DEV_MODE=${VITE_DEV_MODE}

RUN npm run build

# ------------------------------------------------------------------------- runtime
# nginx-unprivileged runs the worker *and* the master as uid 101 — no root anywhere in
# the running container.
FROM nginxinc/nginx-unprivileged:${NGINX_VERSION}-alpine AS runtime

# Defaults match docker-compose (api on the compose network, published on port 80).
# containerapps.bicep overrides both for the deployed environment.
ENV NGINX_PORT=80 \
    API_UPSTREAM=http://api:8000

# /etc/nginx/templates/*.template is rendered with envsubst into /etc/nginx/conf.d/ by
# the base image's entrypoint, so ${API_UPSTREAM} and ${NGINX_PORT} resolve at startup.
COPY nginx.conf /etc/nginx/templates/default.conf.template

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

STOPSIGNAL SIGQUIT
