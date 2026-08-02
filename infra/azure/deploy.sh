#!/usr/bin/env bash
# ---------------------------------------------------------------------------------
# deploy.sh — idempotent deployment of the productionizing-rag-azure Azure platform.
#
#   ./infra/azure/deploy.sh --env dev  --resource-group rg-rag-dev  --what-if
#   ./infra/azure/deploy.sh --env dev  --resource-group rg-rag-dev
#   ./infra/azure/deploy.sh --env prod --resource-group rg-rag-prod --acr myacr
#
# What it does, in order:
#   1. Preflight: az CLI, bicep CLI, an active login, the subscription, the parameter
#      file, the container images, and the Entra identifiers.
#   2. Resolves the deploying principal's object id so Key Vault Secrets Officer can be
#      granted to it (skip with --no-deployer-grant).
#   3. Reuses the Qdrant API key and the PostgreSQL admin password already stored in the
#      environment's Key Vault; generates them only on the very first deployment. This
#      is what makes re-running safe: a redeploy never rotates a credential by accident.
#   4. Runs `az deployment group what-if` or `create` against a *stable* deployment name
#      so repeated runs update one deployment record instead of piling up history.
#   5. Prints the outputs the app and the operational scripts need.
#
# No secret is ever echoed. Values are passed to Bicep through the environment, which
# params/*.bicepparam read with readEnvironmentVariable().
# ---------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENVIRONMENT=""
RESOURCE_GROUP=""
LOCATION="${AZURE_LOCATION:-westeurope}"
SUBSCRIPTION="${AZURE_SUBSCRIPTION_ID:-}"
WHAT_IF="false"
ASSUME_YES="false"
GRANT_DEPLOYER="true"
ROTATE_GENERATED_SECRETS="false"

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_INFO=$'\033[36m'; C_OK=$'\033[32m'
  C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""; C_BOLD=""
fi

log()  { printf '%s==>%s %s\n' "${C_INFO}" "${C_RESET}" "$*"; }
ok()   { printf '%s  ok%s %s\n' "${C_OK}" "${C_RESET}" "$*"; }
warn() { printf '%swarn%s %s\n' "${C_WARN}" "${C_RESET}" "$*" >&2; }
die()  { printf '%sfail%s %s\n' "${C_ERR}" "${C_RESET}" "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: deploy.sh --env <dev|prod> --resource-group <name> [options]

Required:
  -e, --env NAME              Parameter set to use: dev or prod (infra/azure/params/NAME.bicepparam)
  -g, --resource-group NAME   Target resource group (created when missing)

Options:
  -l, --location NAME         Azure region for the resource group        (default: $AZURE_LOCATION or westeurope)
  -s, --subscription ID       Subscription id or name to deploy into     (default: current az account)
      --api-image REF         API image reference                        (env: API_IMAGE)
      --web-image REF         Web image reference                        (env: WEB_IMAGE)
      --acr NAME              ACR in this resource group; grants AcrPull and derives the login server
      --acr-login-server FQDN ACR login server when the registry is elsewhere
      --what-if               Show the change preview and exit without deploying
      --rotate-secrets        Regenerate the Qdrant API key and PostgreSQL password (breaks running apps until redeploy)
      --no-deployer-grant     Do not grant Key Vault Secrets Officer to the deploying principal
  -y, --yes                   Do not prompt before deploying
  -h, --help                  Show this help

Environment variables read (see params/*.bicepparam):
  ANTHROPIC_API_KEY (required)   ENTRA_TENANT_ID (required)   ENTRA_CLIENT_ID (required)
  ENTRA_AUDIENCE   LANGFUSE_HOST   LANGFUSE_PUBLIC_KEY   LANGFUSE_SECRET_KEY   PII_HASH_SECRET
  QDRANT_API_KEY   POSTGRES_ADMIN_PASSWORD   (resolved from Key Vault when unset)
  QDRANT_IMAGE     RAG_INGEST_CRON   RAG_INGEST_TIMEZONE   RAG_LOG_LEVEL   RAG_BASE_NAME
  POSTGRES_ENTRA_ADMIN_OBJECT_ID   POSTGRES_ENTRA_ADMIN_PRINCIPAL_NAME
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env)               ENVIRONMENT="${2:?}"; shift 2 ;;
    -g|--resource-group)    RESOURCE_GROUP="${2:?}"; shift 2 ;;
    -l|--location)          LOCATION="${2:?}"; shift 2 ;;
    -s|--subscription)      SUBSCRIPTION="${2:?}"; shift 2 ;;
    --api-image)            export API_IMAGE="${2:?}"; shift 2 ;;
    --web-image)            export WEB_IMAGE="${2:?}"; shift 2 ;;
    --acr)                  export ACR_NAME="${2:?}"; shift 2 ;;
    --acr-login-server)     export ACR_LOGIN_SERVER="${2:?}"; shift 2 ;;
    --what-if)              WHAT_IF="true"; shift ;;
    --rotate-secrets)       ROTATE_GENERATED_SECRETS="true"; shift ;;
    --no-deployer-grant)    GRANT_DEPLOYER="false"; shift ;;
    -y|--yes)               ASSUME_YES="true"; shift ;;
    -h|--help)              usage; exit 0 ;;
    *)                      usage; die "unknown argument: $1" ;;
  esac
done

[[ -n "${ENVIRONMENT}" ]]    || { usage; die "--env is required"; }
[[ -n "${RESOURCE_GROUP}" ]] || { usage; die "--resource-group is required"; }

PARAM_FILE="${SCRIPT_DIR}/params/${ENVIRONMENT}.bicepparam"
TEMPLATE_FILE="${SCRIPT_DIR}/main.bicep"
DEPLOYMENT_NAME="rag-${ENVIRONMENT}"

# ------------------------------------------------------------------------ preflight
log "preflight"

command -v az >/dev/null 2>&1 || die "the Azure CLI (az) is not on PATH"
[[ -f "${PARAM_FILE}" ]] || die "parameter file not found: ${PARAM_FILE}"
[[ -f "${TEMPLATE_FILE}" ]] || die "template not found: ${TEMPLATE_FILE}"

if ! az bicep version >/dev/null 2>&1; then
  log "installing the bicep CLI"
  az bicep install >/dev/null || die "could not install the bicep CLI"
fi
ok "bicep $(az bicep version 2>/dev/null | head -n1)"

if ! az account show >/dev/null 2>&1; then
  die "not signed in: run 'az login' (or 'az login --identity' on a runner)"
fi

if [[ -n "${SUBSCRIPTION}" ]]; then
  az account set --subscription "${SUBSCRIPTION}" || die "cannot select subscription ${SUBSCRIPTION}"
fi
SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
SUBSCRIPTION_NAME="$(az account show --query name -o tsv)"
ok "subscription ${SUBSCRIPTION_NAME} (${SUBSCRIPTION_ID})"

export AZURE_LOCATION="${LOCATION}"

# Required inputs the template cannot invent.
[[ -n "${ANTHROPIC_API_KEY:-}" ]] || die "ANTHROPIC_API_KEY is not set; the platform cannot answer without it"
[[ -n "${ENTRA_TENANT_ID:-}" ]]   || die "ENTRA_TENANT_ID is not set; token validation needs the directory id"
[[ -n "${ENTRA_CLIENT_ID:-}" ]]   || die "ENTRA_CLIENT_ID is not set; token validation needs the expected audience"
[[ -n "${API_IMAGE:-}" ]]         || die "API_IMAGE is not set (use --api-image); build and push it first"
[[ -n "${WEB_IMAGE:-}" ]]         || die "WEB_IMAGE is not set (use --web-image); build and push it first"

if [[ -n "${ACR_NAME:-}" && -z "${ACR_LOGIN_SERVER:-}" ]]; then
  ACR_LOGIN_SERVER="$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv 2>/dev/null || true)"
  [[ -n "${ACR_LOGIN_SERVER}" ]] || die "ACR '${ACR_NAME}' not found in subscription ${SUBSCRIPTION_ID}"
  export ACR_LOGIN_SERVER
fi
if [[ -n "${ACR_LOGIN_SERVER:-}" ]]; then
  ok "registry ${ACR_LOGIN_SERVER}"
else
  warn "no container registry supplied: ${API_IMAGE} and ${WEB_IMAGE} must be publicly pullable"
fi

if [[ -n "${LANGFUSE_PUBLIC_KEY:-}" || -n "${LANGFUSE_SECRET_KEY:-}" ]]; then
  [[ -n "${LANGFUSE_PUBLIC_KEY:-}" && -n "${LANGFUSE_SECRET_KEY:-}" ]] \
    || die "Langfuse needs both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or neither"
  [[ -n "${LANGFUSE_HOST:-}" ]] || die "LANGFUSE_HOST is required when the Langfuse keys are set"
  ok "Langfuse tracing enabled against ${LANGFUSE_HOST}"
else
  warn "Langfuse keys absent: tracing degrades to the no-op tracer"
fi

# ------------------------------------------------------- deploying principal object id
if [[ "${GRANT_DEPLOYER}" == "true" && -z "${DEPLOYER_PRINCIPAL_ID:-}" ]]; then
  principal_type="$(az account show --query user.type -o tsv 2>/dev/null || echo user)"
  if [[ "${principal_type}" == "servicePrincipal" ]]; then
    client_id="$(az account show --query user.name -o tsv)"
    DEPLOYER_PRINCIPAL_ID="$(az ad sp show --id "${client_id}" --query id -o tsv 2>/dev/null || true)"
    DEPLOYER_PRINCIPAL_TYPE="ServicePrincipal"
  else
    DEPLOYER_PRINCIPAL_ID="$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)"
    DEPLOYER_PRINCIPAL_TYPE="User"
  fi
  if [[ -n "${DEPLOYER_PRINCIPAL_ID}" ]]; then
    export DEPLOYER_PRINCIPAL_ID DEPLOYER_PRINCIPAL_TYPE
    ok "deploying principal ${DEPLOYER_PRINCIPAL_ID} (${DEPLOYER_PRINCIPAL_TYPE}) will get Key Vault Secrets Officer"
  else
    warn "could not resolve the deploying principal's object id; skipping the Key Vault grant"
  fi
fi

# ------------------------------------------------------------------- resource group
if az group show --name "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  ok "resource group ${RESOURCE_GROUP} exists"
else
  log "creating resource group ${RESOURCE_GROUP} in ${LOCATION}"
  az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output none
  ok "resource group created"
fi

# --------------------------------------------------------------- generated secrets
# Reuse whatever is already in the environment's Key Vault so a redeploy is a no-op for
# credentials. Only the first deployment generates them.
resolve_key_vault_name() {
  local name
  name="$(az deployment group show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${DEPLOYMENT_NAME}" \
    --query 'properties.outputs.keyVaultName.value' \
    -o tsv 2>/dev/null || true)"
  if [[ -z "${name}" || "${name}" == "null" ]]; then
    name="$(az keyvault list \
      --resource-group "${RESOURCE_GROUP}" \
      --query "[?starts_with(name, 'kv-')].name | [0]" \
      -o tsv 2>/dev/null || true)"
  fi
  [[ "${name}" == "null" ]] && name=""
  printf '%s' "${name}"
}

read_vault_secret() {
  local vault="$1" secret="$2"
  [[ -n "${vault}" ]] || return 0
  az keyvault secret show --vault-name "${vault}" --name "${secret}" \
    --query value -o tsv 2>/dev/null || true
}

random_secret() {
  # 32 bytes of entropy, URL-safe, no shell-hostile characters.
  local raw
  raw="$(openssl rand -base64 48 2>/dev/null | tr -d '\n=+/' | cut -c1-40)"
  [[ -n "${raw}" ]] || raw="$(head -c 512 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-40)"
  printf '%s' "${raw}"
}

KEY_VAULT_NAME="$(resolve_key_vault_name)"
if [[ -n "${KEY_VAULT_NAME}" ]]; then
  ok "existing key vault ${KEY_VAULT_NAME}"
else
  log "no key vault found yet: this is a first deployment"
fi

if [[ "${ROTATE_GENERATED_SECRETS}" == "true" ]]; then
  warn "--rotate-secrets: generating a new Qdrant API key and PostgreSQL password"
  QDRANT_API_KEY="$(random_secret)"
  POSTGRES_ADMIN_PASSWORD="$(random_secret)"
else
  if [[ -z "${QDRANT_API_KEY:-}" ]]; then
    QDRANT_API_KEY="$(read_vault_secret "${KEY_VAULT_NAME}" qdrant-api-key)"
    if [[ -n "${QDRANT_API_KEY}" ]]; then
      ok "reusing the stored Qdrant API key"
    fi
  fi
  if [[ -z "${POSTGRES_ADMIN_PASSWORD:-}" ]]; then
    POSTGRES_ADMIN_PASSWORD="$(
      read_vault_secret "${KEY_VAULT_NAME}" postgres-admin-password
    )"
    if [[ -n "${POSTGRES_ADMIN_PASSWORD}" ]]; then
      ok "reusing the stored PostgreSQL password"
    fi
  fi
  if [[ -z "${QDRANT_API_KEY}" ]]; then
    QDRANT_API_KEY="$(random_secret)"
    log "generated a Qdrant API key"
  fi
  if [[ -z "${POSTGRES_ADMIN_PASSWORD}" ]]; then
    POSTGRES_ADMIN_PASSWORD="$(random_secret)"
    log "generated a PostgreSQL password"
  fi
fi
export QDRANT_API_KEY POSTGRES_ADMIN_PASSWORD

# ------------------------------------------------------------------------ template
log "linting ${TEMPLATE_FILE#"${REPO_ROOT}"/}"
az bicep build --file "${TEMPLATE_FILE}" --stdout >/dev/null || die "bicep build failed"
ok "template compiles"

# --------------------------------------------------------------------------- deploy
if [[ "${WHAT_IF}" == "true" ]]; then
  log "what-if against ${RESOURCE_GROUP} (nothing will be changed)"
  az deployment group what-if \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${DEPLOYMENT_NAME}" \
    --parameters "${PARAM_FILE}"
  exit 0
fi

if [[ "${ASSUME_YES}" != "true" ]]; then
  printf '%sDeploy %s to %s/%s? [y/N] %s' \
    "${C_BOLD}" "${ENVIRONMENT}" "${SUBSCRIPTION_NAME}" "${RESOURCE_GROUP}" "${C_RESET}"
  read -r reply
  [[ "${reply}" =~ ^[Yy]$ ]] || die "aborted"
fi

log "deploying ${DEPLOYMENT_NAME}"
az deployment group create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${DEPLOYMENT_NAME}" \
  --parameters "${PARAM_FILE}" \
  --output none
ok "deployment complete"

# -------------------------------------------------------------------------- outputs
log "outputs"
az deployment group show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${DEPLOYMENT_NAME}" \
  --query 'properties.outputs' \
  --output json

API_URL="$(az deployment group show -g "${RESOURCE_GROUP}" -n "${DEPLOYMENT_NAME}" \
  --query 'properties.outputs.apiUrl.value' -o tsv)"
WEB_URL="$(az deployment group show -g "${RESOURCE_GROUP}" -n "${DEPLOYMENT_NAME}" \
  --query 'properties.outputs.webUrl.value' -o tsv)"
FUNCTION_APP="$(az deployment group show -g "${RESOURCE_GROUP}" -n "${DEPLOYMENT_NAME}" \
  --query 'properties.outputs.functionAppName.value' -o tsv)"

cat <<NEXT

${C_BOLD}Next steps${C_RESET}
  1. Apply migrations (from a host inside the VNet, or over a jump box):
       uv run alembic -c packages/ragcore/ragcore/db/alembic.ini upgrade head
  2. Create the Qdrant collections and payload indexes:
       uv run python scripts/bootstrap_qdrant.py
  3. Publish the ingestion Function App:
       func azure functionapp publish ${FUNCTION_APP} --python
  4. Smoke test the deployment:
       uv run python scripts/smoke_test.py --base-url ${API_URL}

  api  ${API_URL}
  web  ${WEB_URL}
NEXT
