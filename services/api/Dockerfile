# ---------------------------------------------------------------------------------
# services/api/Dockerfile — FastAPI image for the RAG API.
#
# Build context is the **repository root** (docker-compose.yml sets `context: .`) so the
# build can copy `packages/ragcore`, which is a uv workspace path dependency.
#
#   docker build -f services/api/Dockerfile -t rag-api:dev .
#
# Two stages: a builder that resolves the uv workspace into /opt/venv, and a slim
# runtime that copies only that virtual environment. The API package is installed
# non-editable, so no source tree is needed at runtime.
#
# The image runs as an unprivileged user (uid 10001). FastEmbed model weights land in
# $RAG_EMBEDDING_CACHE_DIR under that user's home; mount a volume there (compose does)
# so a restart does not re-download bge-m3 and the reranker.
# ---------------------------------------------------------------------------------

# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.5.11

# ------------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG UV_VERSION

# UV_PYTHON pins the build to the base image's interpreter rather than a uv-downloaded
# one: the runtime stage shares this base, so /opt/venv/bin/python resolves there too.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.13 \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "uv==${UV_VERSION}"

WORKDIR /src

# uv resolves the whole workspace, so every member's manifest has to be present even
# though only ragcore and the api package are installed. `uv.lock*` is a tolerant glob:
# the COPY succeeds because pyproject.toml always matches, and the lock is used when it
# has been committed.
COPY pyproject.toml README.md uv.lock* ./
COPY packages/ragcore/pyproject.toml ./packages/ragcore/
COPY services/api/pyproject.toml ./services/api/
COPY services/ingestion/pyproject.toml ./services/ingestion/
COPY services/eval/pyproject.toml ./services/eval/

COPY packages/ragcore ./packages/ragcore
COPY services/api ./services/api

# The api distribution name is read from its own manifest rather than hard-coded, so a
# rename in services/api/pyproject.toml cannot silently break the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    API_PACKAGE="$(python -c "import pathlib, tomllib; print(tomllib.loads(pathlib.Path('services/api/pyproject.toml').read_text())['project']['name'])")" \
    && echo "installing workspace package: ${API_PACKAGE}" \
    && uv sync --no-dev --no-editable --package "${API_PACKAGE}"

# ------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    RAG_EMBEDDING_CACHE_DIR=/home/app/.cache/fastembed

# libgomp1 is required by onnxruntime, which FastEmbed uses for bge-m3 and the
# cross-encoder reranker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

RUN mkdir -p /home/app/.cache/fastembed /home/app/.cache/huggingface \
    && chown -R 10001:10001 /home/app

USER 10001:10001
WORKDIR /home/app

EXPOSE 8000

# No curl or wget in the image: probe with the interpreter that is already here.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD ["python", "-c", "import os,sys,urllib.request;port=os.environ.get('RAG_API_PORT','8000');sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=5).status==200 else 1)"]

# RAG_API_HOST / RAG_API_PORT are ragcore.settings fields; the defaults here mirror
# their defaults so nothing is hard-coded in two places with different values.
CMD ["sh", "-c", "exec uvicorn app.main:app --host \"${RAG_API_HOST:-0.0.0.0}\" --port \"${RAG_API_PORT:-8000}\" --proxy-headers --forwarded-allow-ips '*' --no-server-header"]
