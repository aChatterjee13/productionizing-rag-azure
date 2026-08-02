// ---------------------------------------------------------------------------------
// containerapps.bicep — the shared managed environment plus the api and web apps.
//
// The environment is created here because Qdrant, the API and the web app all live in
// it. qdrant.bicep consumes `environmentId` from this module's outputs, and this
// module computes Qdrant's internal address from `qdrantAppName` plus the environment's
// own defaultDomain — that is what keeps the dependency a straight line
// (containerapps -> qdrant) instead of a cycle.
//
// Every secret arrives as a Key Vault reference resolved by the app's user-assigned
// identity (`secrets[].keyVaultUrl` + `secrets[].identity`). No secret value is ever a
// parameter of this module.
// ---------------------------------------------------------------------------------

@description('Container Apps managed environment name.')
param environmentName string

@description('Azure region.')
param location string

@description('Tags applied to every resource in this module.')
param tags object = {}

@description('Log Analytics workspace the environment streams console and system logs to.')
param logAnalyticsWorkspaceName string

@description('Infrastructure subnet for the environment. Must be delegated to Microsoft.App/environments.')
param infrastructureSubnetId string

@description('Spread replicas across availability zones. Requires three zones in the region.')
param zoneRedundant bool = false

// ------------------------------------------------------------------------ app names
@description('Name of the API container app.')
param apiAppName string

@description('Name of the web container app.')
param webAppName string

@description('Name of the Qdrant container app; used to derive its internal address.')
param qdrantAppName string

// --------------------------------------------------------------------------- images
@description('Fully qualified API image reference.')
param apiImage string

@description('Fully qualified web image reference.')
param webImage string

@description('ACR login server for image pulls. Empty means the images are public.')
param containerRegistryLoginServer string = ''

// ------------------------------------------------------------------------ identities
@description('Resource id of the API user-assigned identity.')
param apiIdentityId string

@description('Client id of the API user-assigned identity (RAG_AZURE_CLIENT_ID).')
param apiIdentityClientId string

@description('Resource id of the web user-assigned identity.')
param webIdentityId string

// ----------------------------------------------------------------------- api sizing
@description('API CPU cores, as a string so it can be passed to json().')
param apiCpu string = '1.0'

@description('API memory, e.g. 2Gi. Container Apps requires memory = 2x CPU in GiB.')
param apiMemory string = '2Gi'

@description('Minimum API replicas. Keep at least 1 so FastEmbed models stay warm.')
@minValue(1)
param apiMinReplicas int = 1

@description('Maximum API replicas.')
@minValue(1)
param apiMaxReplicas int = 5

@description('Concurrent requests per replica before the HTTP scaler adds one.')
param apiConcurrentRequests int = 20

@description('Port the API listens on inside the container.')
param apiTargetPort int = 8000

// ----------------------------------------------------------------------- web sizing
@description('Web CPU cores, as a string.')
param webCpu string = '0.25'

@description('Web memory.')
param webMemory string = '0.5Gi'

@description('Minimum web replicas.')
@minValue(1)
param webMinReplicas int = 1

@description('Maximum web replicas.')
@minValue(1)
param webMaxReplicas int = 3

@description('Port nginx listens on inside the web container.')
param webTargetPort int = 80

// ------------------------------------------------------------------- app configuration
@description('Value of RAG_ENV inside the containers.')
@allowed([
  'local'
  'dev'
  'staging'
  'production'
])
param ragEnv string

@description('Root log level.')
@allowed([
  'DEBUG'
  'INFO'
  'WARNING'
  'ERROR'
  'CRITICAL'
])
param logLevel string = 'INFO'

@description('PostgreSQL host (RAG_POSTGRES_HOST).')
param postgresHost string

@description('PostgreSQL user (RAG_POSTGRES_USER).')
param postgresUser string

@description('PostgreSQL database (RAG_POSTGRES_DB).')
param postgresDatabase string

@description('Blob service endpoint (RAG_AZURE_BLOB_ACCOUNT_URL).')
param blobAccountUrl string

@description('Container holding source documents (RAG_AZURE_BLOB_CONTAINER).')
param sourcesContainerName string

@description('Container holding archived raw bytes (RAG_AZURE_BLOB_RAW_CONTAINER).')
param rawContainerName string

@description('Container holding ingest manifests (RAG_INGEST_MANIFEST_CONTAINER).')
param manifestsContainerName string

@description('Ingest queue name (RAG_AZURE_STORAGE_QUEUE_NAME).')
param ingestQueueName string

@description('Key Vault URI (RAG_AZURE_KEY_VAULT_URL).')
param keyVaultUri string

@description('Versionless Key Vault secret URIs, from keyvault.bicep.')
param secretUris object

@description('Entra directory (tenant) id.')
param entraTenantId string

@description('Entra API application (client) id.')
param entraClientId string

@description('Explicit expected audience. Empty falls back to the client id.')
param entraAudience string = ''

@description('Enable Langfuse tracing. Requires both Langfuse keys in Key Vault.')
param langfuseEnabled bool = false

@description('Langfuse base URL.')
param langfuseHost string = ''

@description('Store a Key Vault reference for the PII HMAC key.')
param piiHashSecretConfigured bool = false

@description('Six-field NCRONTAB ingestion schedule (RAG_INGEST_CRON).')
param ingestCron string = '0 30 2 * * *'

@description('IANA timezone for the ingest schedule and the working-hours guard.')
param ingestTimezone string = 'UTC'

@description('First hour of the working day.')
param ingestWorkingHoursStart int = 8

@description('End hour of the working day, exclusive.')
param ingestWorkingHoursEnd int = 18

@description('Browser origins the API accepts, as a JSON array string. Empty derives the web app origin.')
param apiCorsOrigins string = ''

@description('Extra environment variables appended to the API container.')
param additionalApiEnv array = []

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: logAnalyticsWorkspaceName
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: infrastructureSubnetId
      internal: false
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    zoneRedundant: zoneRedundant
  }
}

var defaultDomain = managedEnvironment.properties.defaultDomain
var apiFqdn = '${apiAppName}.${defaultDomain}'
var webFqdn = '${webAppName}.${defaultDomain}'

// Internal ingress resolves on <app>.internal.<defaultDomain>; the ingress maps port 80
// onto Qdrant's targetPort, so the URL carries no port.
var qdrantInternalUrl = 'http://${qdrantAppName}.internal.${defaultDomain}'

var resolvedCorsOrigins = empty(apiCorsOrigins) ? '["https://${webFqdn}"]' : apiCorsOrigins

var registries = empty(containerRegistryLoginServer)
  ? []
  : [
      {
        server: containerRegistryLoginServer
        identity: apiIdentityId
      }
    ]

var webRegistries = empty(containerRegistryLoginServer)
  ? []
  : [
      {
        server: containerRegistryLoginServer
        identity: webIdentityId
      }
    ]

var baseApiSecrets = [
  {
    name: 'anthropic-api-key'
    keyVaultUrl: secretUris.anthropicApiKey
    identity: apiIdentityId
  }
  {
    name: 'qdrant-api-key'
    keyVaultUrl: secretUris.qdrantApiKey
    identity: apiIdentityId
  }
  {
    name: 'postgres-password'
    keyVaultUrl: secretUris.postgresAdminPassword
    identity: apiIdentityId
  }
]

var langfuseSecrets = langfuseEnabled
  ? [
      {
        name: 'langfuse-public-key'
        keyVaultUrl: secretUris.langfusePublicKey
        identity: apiIdentityId
      }
      {
        name: 'langfuse-secret-key'
        keyVaultUrl: secretUris.langfuseSecretKey
        identity: apiIdentityId
      }
    ]
  : []

var piiSecrets = piiHashSecretConfigured
  ? [
      {
        name: 'pii-hash-secret'
        keyVaultUrl: secretUris.piiHashSecret
        identity: apiIdentityId
      }
    ]
  : []

var baseApiEnv = [
  {
    name: 'RAG_ENV'
    value: ragEnv
  }
  {
    name: 'RAG_SERVICE_NAME'
    value: 'rag-api'
  }
  {
    name: 'RAG_LOG_LEVEL'
    value: logLevel
  }
  {
    name: 'RAG_LOG_JSON'
    value: 'true'
  }
  {
    name: 'RAG_QDRANT_URL'
    value: qdrantInternalUrl
  }
  {
    name: 'RAG_QDRANT_API_KEY'
    secretRef: 'qdrant-api-key'
  }
  {
    name: 'RAG_POSTGRES_HOST'
    value: postgresHost
  }
  {
    name: 'RAG_POSTGRES_PORT'
    value: '5432'
  }
  {
    name: 'RAG_POSTGRES_USER'
    value: postgresUser
  }
  {
    name: 'RAG_POSTGRES_PASSWORD'
    secretRef: 'postgres-password'
  }
  {
    name: 'RAG_POSTGRES_DB'
    value: postgresDatabase
  }
  {
    name: 'RAG_POSTGRES_SSLMODE'
    value: 'require'
  }
  {
    // No Azure Cache for Redis is provisioned; rate limiting and SSE fan-out degrade
    // to in-process behaviour rather than failing.
    name: 'RAG_REDIS_ENABLED'
    value: 'false'
  }
  {
    name: 'RAG_ANTHROPIC_API_KEY'
    secretRef: 'anthropic-api-key'
  }
  {
    name: 'RAG_ENTRA_TENANT_ID'
    value: entraTenantId
  }
  {
    name: 'RAG_ENTRA_CLIENT_ID'
    value: entraClientId
  }
  {
    name: 'RAG_ENTRA_AUDIENCE'
    value: entraAudience
  }
  {
    name: 'RAG_ENTRA_DEV_MODE'
    value: 'false'
  }
  {
    name: 'RAG_AZURE_BLOB_ACCOUNT_URL'
    value: blobAccountUrl
  }
  {
    name: 'RAG_AZURE_BLOB_CONTAINER'
    value: sourcesContainerName
  }
  {
    name: 'RAG_AZURE_BLOB_RAW_CONTAINER'
    value: rawContainerName
  }
  {
    name: 'RAG_INGEST_MANIFEST_CONTAINER'
    value: manifestsContainerName
  }
  {
    name: 'RAG_AZURE_STORAGE_QUEUE_NAME'
    value: ingestQueueName
  }
  {
    name: 'RAG_AZURE_KEY_VAULT_URL'
    value: keyVaultUri
  }
  {
    name: 'RAG_AZURE_USE_MANAGED_IDENTITY'
    value: 'true'
  }
  {
    name: 'RAG_AZURE_CLIENT_ID'
    value: apiIdentityClientId
  }
  {
    name: 'RAG_INGEST_CRON'
    value: ingestCron
  }
  {
    name: 'RAG_INGEST_TIMEZONE'
    value: ingestTimezone
  }
  {
    name: 'RAG_INGEST_WORKING_HOURS_START'
    value: string(ingestWorkingHoursStart)
  }
  {
    name: 'RAG_INGEST_WORKING_HOURS_END'
    value: string(ingestWorkingHoursEnd)
  }
  {
    name: 'RAG_EMBEDDING_CACHE_DIR'
    value: '/home/app/.cache/fastembed'
  }
  {
    name: 'RAG_API_HOST'
    value: '0.0.0.0'
  }
  {
    name: 'RAG_API_PORT'
    value: string(apiTargetPort)
  }
  {
    name: 'RAG_API_CORS_ORIGINS'
    value: resolvedCorsOrigins
  }
  {
    name: 'RAG_LANGFUSE_ENABLED'
    value: langfuseEnabled ? 'true' : 'false'
  }
]

var langfuseEnv = langfuseEnabled
  ? [
      {
        name: 'RAG_LANGFUSE_HOST'
        value: langfuseHost
      }
      {
        name: 'RAG_LANGFUSE_PUBLIC_KEY'
        secretRef: 'langfuse-public-key'
      }
      {
        name: 'RAG_LANGFUSE_SECRET_KEY'
        secretRef: 'langfuse-secret-key'
      }
    ]
  : []

var piiEnv = piiHashSecretConfigured
  ? [
      {
        name: 'RAG_PII_HASH_SECRET'
        secretRef: 'pii-hash-secret'
      }
    ]
  : []

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiAppName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${apiIdentityId}': {}
    }
  }
  properties: {
    environmentId: managedEnvironment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: apiTargetPort
        transport: 'auto'
        allowInsecure: false
        clientCertificateMode: 'ignore'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: registries
      secrets: concat(baseApiSecrets, langfuseSecrets, piiSecrets)
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: {
            cpu: json(apiCpu)
            memory: apiMemory
          }
          env: concat(baseApiEnv, langfuseEnv, piiEnv, additionalApiEnv)
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: apiTargetPort
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              // FastEmbed downloads bge-m3 and the reranker on first boot.
              failureThreshold: 60
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/readyz'
                port: apiTargetPort
                scheme: 'HTTP'
              }
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 6
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: apiTargetPort
                scheme: 'HTTP'
              }
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas
        maxReplicas: apiMaxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: string(apiConcurrentRequests)
              }
            }
          }
        ]
      }
    }
  }
}

resource webApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: webAppName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${webIdentityId}': {}
    }
  }
  properties: {
    environmentId: managedEnvironment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: webTargetPort
        transport: 'auto'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: webRegistries
      secrets: []
    }
    template: {
      containers: [
        {
          name: 'web'
          image: webImage
          resources: {
            cpu: json(webCpu)
            memory: webMemory
          }
          env: [
            {
              // Substituted into nginx.conf by the nginx image's envsubst entrypoint.
              name: 'API_UPSTREAM'
              value: 'https://${apiFqdn}'
            }
            {
              name: 'NGINX_PORT'
              value: string(webTargetPort)
            }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: {
                path: '/'
                port: webTargetPort
                scheme: 'HTTP'
              }
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 6
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/'
                port: webTargetPort
                scheme: 'HTTP'
              }
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: webMinReplicas
        maxReplicas: webMaxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

@description('Managed environment resource id, consumed by qdrant.bicep.')
output environmentId string = managedEnvironment.id

@description('Managed environment name.')
output environmentName string = managedEnvironment.name

@description('Environment default domain, e.g. bravesea-1a2b3c4d.westeurope.azurecontainerapps.io.')
output defaultDomain string = defaultDomain

@description('Public API FQDN.')
output apiFqdn string = apiFqdn

@description('Public API base URL.')
output apiUrl string = 'https://${apiFqdn}'

@description('Public web FQDN.')
output webFqdn string = webFqdn

@description('Public web base URL.')
output webUrl string = 'https://${webFqdn}'

@description('Internal Qdrant URL other apps use (RAG_QDRANT_URL).')
output qdrantInternalUrl string = qdrantInternalUrl

@description('API container app name.')
output apiAppName string = apiApp.name

@description('Web container app name.')
output webAppName string = webApp.name
