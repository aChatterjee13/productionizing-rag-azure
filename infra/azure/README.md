# Azure infrastructure

Bicep for the whole `productionizing-rag` platform, deployed as **one resource-group
deployment**. Everything is parameterised, nothing carries a secret value, and every
workload authenticates with a managed identity wherever Azure allows it.

```
infra/azure/
├── main.bicep                  composition + naming + outputs
├── deploy.sh                   idempotent deploy / --what-if wrapper
├── params/dev.bicepparam       dev sizing, reads secrets from the environment
├── params/prod.bicepparam      prod sizing (HA, purge protection, zone redundancy)
└── modules/
    ├── network.bicep           VNet + 4 subnets + PostgreSQL private DNS zone
    ├── observability.bicep     Log Analytics + App Insights + diagnostic settings
    ├── storage.bicep           blob containers, ingest queue, Qdrant file share
    ├── keyvault.bicep          RBAC vault + conditional secrets + secret URIs
    ├── identity.bicep          4 user-assigned identities + every role assignment
    ├── postgres.bicep          Flexible Server, private access, Entra admin, backups
    ├── containerapps.bicep     managed environment + api + web container apps
    ├── qdrant.bicep            Qdrant container app with Azure Files persistence
    └── functions.bicep         Flex Consumption Function App (Python 3.13) ingestion
```

## What gets created

| Resource | Name pattern | Notes |
|---|---|---|
| Log Analytics | `log-rag-<env>-<token>` | retention 30d dev / 90d prod |
| Application Insights | `appi-rag-<env>-<token>` | workspace-based |
| Virtual network | `vnet-rag-<env>` | `snet-containerapps` (/23, delegated), `snet-postgres` (delegated), `snet-functions` (delegated), `snet-private-endpoints` |
| Storage account | `strag<env><token>` | containers `rag-documents`, `rag-raw`, `rag-manifests`, `function-releases`, `rag-eval-reports`; queue `rag-ingest` (+ poison); file share `qdrant-storage` |
| Key Vault | `kv-rag<env><token>` | RBAC mode, soft delete, purge protection in prod |
| Managed identities | `id-rag-<env>-{api,web,ingest,qdrant}` | see the RBAC table below |
| PostgreSQL Flexible Server | `psql-rag-<env>-<token>` | v16, **private access only**, Entra admin, 7d/35d backups |
| Container Apps environment | `cae-rag-<env>-<token>` | VNet-integrated, Consumption workload profile, logs to Log Analytics |
| Container apps | `ca-rag-<env>-{api,web,qdrant}` | api/web external ingress, qdrant internal only |
| Function App | `func-rag-<env>-<token>` | Flex Consumption FC1, Python 3.13, Durable task hub |

`<token>` is `uniqueString(subscription().id, resourceGroup().id, environmentName)`, so
names are stable across redeploys and unique across environments.

## Deploying

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ENTRA_TENANT_ID=<directory guid>
export ENTRA_CLIENT_ID=<api app registration client id>

./infra/azure/deploy.sh \
  --env dev --resource-group rg-rag-dev --location westeurope \
  --acr myregistry \
  --api-image myregistry.azurecr.io/rag-api:sha-abc1234 \
  --web-image myregistry.azurecr.io/rag-web:sha-abc1234 \
  --what-if          # drop --what-if to actually deploy
```

`deploy.sh` is safe to re-run. On the first run it generates the Qdrant API key and the
PostgreSQL admin password; on later runs it **reads them back out of Key Vault** and
passes the same values, so a redeploy never silently rotates a credential. Use
`--rotate-secrets` when you actually want new ones.

After the deployment:

```bash
# 1. migrations — from inside the VNet (the server has no public endpoint)
uv run alembic -c packages/ragcore/ragcore/db/alembic.ini upgrade head
# 2. Qdrant collections + payload indexes
uv run python scripts/bootstrap_qdrant.py
# 3. ingestion code
func azure functionapp publish func-rag-dev-<token> --python
# 4. prove it works end to end, ACL negatives included
uv run python scripts/smoke_test.py --base-url https://ca-rag-dev-api.<domain>
```

## Secrets

Six values are secret. They arrive as `@secure()` parameters that `params/*.bicepparam`
reads from the environment with `readEnvironmentVariable`, get written to Key Vault, and
are read back by the workloads through their managed identities:

| Key Vault secret | Environment variable | Consumed as |
|---|---|---|
| `anthropic-api-key` | `ANTHROPIC_API_KEY` | `RAG_ANTHROPIC_API_KEY` |
| `qdrant-api-key` | `QDRANT_API_KEY` (generated) | `QDRANT__SERVICE__API_KEY`, `RAG_QDRANT_API_KEY` |
| `postgres-admin-password` | `POSTGRES_ADMIN_PASSWORD` (generated) | `RAG_POSTGRES_PASSWORD` |
| `langfuse-public-key` | `LANGFUSE_PUBLIC_KEY` | `RAG_LANGFUSE_PUBLIC_KEY` |
| `langfuse-secret-key` | `LANGFUSE_SECRET_KEY` | `RAG_LANGFUSE_SECRET_KEY` |
| `pii-hash-secret` | `PII_HASH_SECRET` | `RAG_PII_HASH_SECRET` |

A secret resource is created **only when its parameter is non-empty**, so omitting an
optional secret on a redeploy leaves the stored value alone instead of overwriting it
with a placeholder. Container Apps reference secrets by versionless URI
(`secrets[].keyVaultUrl` + `secrets[].identity`); the Function App uses
`@Microsoft.KeyVault(SecretUri=…)` with `keyVaultReferenceIdentity`.

No template output contains a secret. The Application Insights connection string is
read inside `functions.bicep` through an `existing` reference rather than passed around.

## RBAC

| Identity | Role | Scope | Why |
|---|---|---|---|
| api | Key Vault Secrets User | vault | Anthropic / Qdrant / PostgreSQL secrets |
| api | Storage Blob Data Reader | account | read archived raw copies and manifests |
| api | Storage Blob Data Contributor | `rag-documents` container | `POST /documents` uploads |
| api | AcrPull | registry | image pull |
| web | AcrPull | registry | image pull |
| ingestion | Key Vault Secrets User | vault | same secrets |
| ingestion | Storage Blob Data Owner | account | Durable task hub blobs + Flex deployment container + manifests |
| ingestion | Storage Queue Data Contributor | account | queue trigger and Durable control queues |
| ingestion | Storage Table Data Contributor | account | Durable history and instance tables |
| qdrant | Key Vault Secrets User | vault | resolve its own API key |
| deployer | Key Vault Secrets Officer | vault | `deploy.sh` seeds and rotates secrets |

Role-assignment names are `guid(scope.id, identity.id, roleDefinitionId)` — seeded with
the identity's *resource id*, not its principal id, because ARM requires an assignment
name to be computable before the deployment starts.

AcrPull is only granted when `containerRegistryName` names a registry **in the same
resource group** — a resource-group deployment cannot write a role assignment in another
group. For a shared registry elsewhere, leave `containerRegistryName` empty, pass
`containerRegistryLoginServer`, and grant AcrPull out of band:

```bash
az role assignment create --assignee <identity clientId> --role AcrPull \
  --scope /subscriptions/<sub>/resourceGroups/<acr-rg>/providers/Microsoft.ContainerRegistry/registries/<acr>
```

## Design decisions worth knowing

**Qdrant persistence is mandatory.** `/qdrant/storage` is an Azure Files volume backed
by the `qdrant-storage` share. Without it a new revision would start with empty
collections and every retrieval would silently return nothing. The mount needs a storage
account key (Container Apps has no managed-identity option for Azure Files), so
`allowSharedKeyAccess` stays `true` on the account and the key is resolved *inside*
`qdrant.bicep` with `listKeys()` — never a parameter, never an output. Every application
data path still uses managed identity.

**Qdrant runs exactly one replica.** `minReplicas` is 1 because a vector database that
scales to zero drops its HNSW index and stalls the next request; `maxReplicas` is also 1
because two replicas mounting the same file share would corrupt the write-ahead log.
Horizontal scale means a real Qdrant cluster with one volume per node, which this
template deliberately does not pretend to do.

**Qdrant is internal-only.** It is reachable at
`http://ca-rag-<env>-qdrant.internal.<defaultDomain>`, which is exactly what
`containerapps.bicep` puts in `RAG_QDRANT_URL`. It has no public FQDN.

**The dependency graph is a line, not a cycle.** The Container Apps environment is
created in `containerapps.bicep`, so `qdrant.bicep` can consume `environmentId`; the API
gets Qdrant's address by deriving it from the app name plus the environment's
`defaultDomain` rather than from Qdrant's outputs. `observability.bicep` is invoked
twice — once with `deployCore: true` for the workspace, once with `deployCore: false`
for the diagnostic settings, which can only be attached after their targets exist.

**The Function App has two identities.** Its data-plane access (storage, Key Vault,
deployment container) uses the pre-provisioned `id-rag-<env>-ingest` user-assigned
identity, because role assignments must exist before the app first starts and a
system-assigned principal id does not exist until after creation. The system-assigned
identity is enabled alongside it for anything that needs the app's own principal.

**No Redis.** `RAG_REDIS_ENABLED=false` is set on both workloads; rate limiting and SSE
fan-out degrade to in-process behaviour. Add an Azure Cache for Redis and flip the
setting if you need cross-replica coordination.

**The web container listens on port 80 as a non-root user.** The image is
`nginx-unprivileged` (uid 101), and both the listen port and the API upstream are runtime
environment variables (`NGINX_PORT`, `API_UPSTREAM`). Binding 80 as non-root needs
`CAP_NET_BIND_SERVICE`, which Docker grants by default. If a platform ever refuses it,
set `webTargetPort: 8080` in the parameter file — `containerapps.bicep` passes that value
through to `NGINX_PORT`, so no image rebuild is needed.

**PostgreSQL has no public endpoint.** Migrations therefore run from inside the VNet — a
Container Apps job, a jump box, or a Bastion session. There is no firewall rule to open,
by design.

## Cost shape (rough, West Europe, dev sizing)

The dev parameter set is deliberately small: burstable PostgreSQL (`Standard_B2s`, 32
GiB), one 1 vCPU / 2 GiB Qdrant replica, 1-2 API replicas, Flex Consumption ingestion
that scales to zero between nightly runs, LRS storage and a 1 GB/day Log Analytics cap.
The two always-on pieces are Qdrant (one replica, by design) and the API minimum
replica; everything else is consumption-priced.

## Local development

There is no Azure dependency for local work — `docker-compose.yml` at the repository
root brings up Qdrant, PostgreSQL, Redis and Langfuse, and `.env.example` documents
every setting. Use this directory only when deploying.
