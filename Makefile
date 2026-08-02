# productionizing-rag — developer entry points.
#
# Docker is not installed on the build machine: `up`/`down` are written to be correct
# and are meant to run on a machine that has Docker. Everything else runs locally
# through uv.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
COMPOSE ?= docker compose
PYTHON ?= $(UV) run python
ALEMBIC_INI := packages/ragcore/ragcore/db/alembic.ini

# Services whose health `up` waits on before returning.
CORE_SERVICES := qdrant postgres redis

.PHONY: help setup up down migrate bootstrap seed api web ingest-local test lint \
        format eval smoke logs ps clean revision downgrade typecheck check

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ environment
setup: ## Create .env, install all workspace members and dev deps
	@test -f .env || { cp .env.example .env; echo "created .env from .env.example"; }
	$(UV) sync --all-packages --all-groups
	@echo
	@echo "Next: make up && make migrate && make bootstrap && make seed"

up: ## Start the local stack (qdrant, postgres, redis, langfuse)
	$(COMPOSE) up -d $(CORE_SERVICES) langfuse-postgres clickhouse minio \
		langfuse-worker langfuse
	@echo "waiting for core services to report healthy..."
	@for svc in $(CORE_SERVICES); do \
		printf '  %-10s' "$$svc"; \
		for i in $$(seq 1 60); do \
			state=$$($(COMPOSE) ps --format '{{.Health}}' $$svc 2>/dev/null || echo ""); \
			if [ "$$state" = "healthy" ]; then echo "healthy"; break; fi; \
			if [ "$$i" = "60" ]; then echo "TIMEOUT ($$state)"; exit 1; fi; \
			sleep 2; \
		done; \
	done
	@echo "qdrant   http://localhost:6333/dashboard"
	@echo "langfuse http://localhost:3000"

down: ## Stop the local stack (keeps volumes)
	$(COMPOSE) down --remove-orphans

ps: ## Show stack status
	$(COMPOSE) ps

logs: ## Tail logs for all services (SERVICE=api to narrow)
	$(COMPOSE) logs -f --tail=100 $(SERVICE)

clean: ## Stop the stack and DELETE all volumes (destroys local data)
	$(COMPOSE) down -v --remove-orphans
	rm -rf .pytest_cache .ruff_cache

# --------------------------------------------------------------------- database
migrate: ## Apply Alembic migrations to head
	$(UV) run alembic -c $(ALEMBIC_INI) upgrade head

downgrade: ## Roll back one migration (REV=-1 by default)
	$(UV) run alembic -c $(ALEMBIC_INI) downgrade $(or $(REV),-1)

revision: ## Autogenerate a migration (M="describe the change")
	@test -n "$(M)" || { echo 'usage: make revision M="add x to y"'; exit 1; }
	$(UV) run alembic -c $(ALEMBIC_INI) revision --autogenerate -m "$(M)"

# ------------------------------------------------------------------- bootstrap
bootstrap: ## Create Qdrant collections + payload indexes, verify connectivity
	$(PYTHON) scripts/bootstrap_qdrant.py

seed: ## Load the sample corpus and demo tenants/users
	$(PYTHON) scripts/seed_demo_tenant.py

# ------------------------------------------------------------------------- run
api: ## Run the API with reload on http://localhost:8000
	$(UV) run uvicorn app.main:app \
		--host $${RAG_API_HOST:-127.0.0.1} \
		--port $${RAG_API_PORT:-8000} \
		--reload \
		--app-dir services/api

web: ## Run the Vite dev server on http://localhost:5173
	cd web && npm install && npm run dev

ingest-local: ## Run one ingestion pass against the local filesystem connector
	$(PYTHON) -m ingestion.cli run \
		--source-type local \
		--tenant $${RAG_SEED_TENANT:-tenant-acme} \
		--force

# ----------------------------------------------------------------------- checks
test: ## Run the test suite
	$(UV) run pytest

lint: ## Lint and check formatting (no writes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-fix lint findings and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# mypy is deliberately absent from the dev dependency group — CI does not run it
# (see README) — so the target brings its own copy rather than failing with
# "No module named mypy" on a freshly synced checkout.
typecheck: ## Type-check the Python packages
	$(UV) run --with mypy mypy \
		packages/ragcore services/api services/ingestion services/eval

check: lint test ## Lint then test — what CI runs

eval: ## Run the golden-set evaluation and apply the CI gate
	$(UV) run python -m eval.run \
		--golden $${RAG_EVAL_GOLDEN_PATH:-services/eval/golden/golden_set.yaml} \
		--gate

smoke: ## End-to-end smoke test against a running API
	$(PYTHON) scripts/smoke_test.py --base-url $${SMOKE_BASE_URL:-http://localhost:8000}
