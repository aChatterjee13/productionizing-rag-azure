// ---------------------------------------------------------------------------------
// main.bicep — the whole productionizing-rag platform in one resource-group deployment.
//
//   observability ──> containerapps ──> qdrant
//        │                 │              │
//        ├──> storage ─────┤              │
//        ├──> keyvault ────┤              │
//        ├──> identity ────┤              │
//        ├──> network ─────┴──> postgres  │
//        └──> functions <─────────────────┘
//        └──> diagnostics (second observability pass, wires the two log sources)
//
// Naming: every resource is `<abbr>-<base>-<env>[-<token>]`, where the token is a
// deterministic `uniqueString(subscription, resourceGroup, environmentName)` so two
// environments in the same subscription never collide and a redeploy is idempotent.
//
// Secrets: only the six @secure() parameters below ever carry a secret value, they are
// written to Key Vault, and every consumer reads them back through a managed identity.
// No output of this template contains a secret.
// ---------------------------------------------------------------------------------

targetScope = 'resourceGroup'

// ------------------------------------------------------------------------- naming
@description('Short base name for every resource. Lowercase alphanumeric, 2-6 characters.')
@minLength(2)
@maxLength(6)
param baseName string = 'rag'

@description('Environment discriminator used in resource names, e.g. dev, stg, prod.')
@minLength(2)
@maxLength(6)
param environmentName string

@description('Azure region for every resource.')
param location string = resourceGroup().location

@description('Tags applied to every resource.')
param tags object = {}

@description('Value of RAG_ENV inside the workloads. Must be "production" for prod.')
@allowed([
  'local'
  'dev'
  'staging'
  'production'
])
param ragEnv string = 'dev'

@description('Root log level for every workload.')
@allowed([
  'DEBUG'
  'INFO'
  'WARNING'
  'ERROR'
  'CRITICAL'
])
param logLevel string = 'INFO'

// ------------------------------------------------------------------------ secrets
@description('Anthropic API key. Stored in Key Vault; workloads read it by identity.')
@secure()
param anthropicApiKey string

@description('Qdrant API key. deploy.sh reuses the stored value or generates one.')
@secure()
param qdrantApiKey string

@description('PostgreSQL administrator password. deploy.sh reuses or generates it.')
@secure()
param postgresAdminPassword string

@description('Langfuse public key. Empty disables Langfuse tracing.')
@secure()
param langfusePublicKey string = ''

@description('Langfuse secret key. Empty disables Langfuse tracing.')
@secure()
param langfuseSecretKey string = ''

@description('HMAC key for hashed PII redaction. Empty leaves the ragcore default.')
@secure()
param piiHashSecret string = ''

@description('Langfuse base URL. Required when the Langfuse keys are supplied.')
param langfuseHost string = ''

// -------------------------------------------------------------------------- entra
@description('Entra directory (tenant) GUID the API validates tokens against.')
param entraTenantId string

@description('Entra API application (client) id; the expected token audience.')
param entraClientId string

@description('Explicit expected audience when tokens use the api:// URI form.')
param entraAudience string = ''

// ------------------------------------------------------------------------- images
@description('Fully qualified API image reference, e.g. myacr.azurecr.io/rag-api:sha-abc123.')
param apiImage string

@description('Fully qualified web image reference.')
param webImage string

@description('Pinned Qdrant image. Never a floating tag for a stateful service.')
param qdrantImage string = 'qdrant/qdrant:v1.12.4'

@description('ACR login server used to pull the api/web images. Empty means public images.')
param containerRegistryLoginServer string = ''

@description('ACR name, when the registry lives in this resource group, so AcrPull can be granted here.')
param containerRegistryName string = ''

// --------------------------------------------------------------------- networking
@description('Virtual network address space.')
param vnetAddressPrefix string = '10.60.0.0/16'

@description('Container Apps infrastructure subnet. Must be /23 or larger.')
param containerAppsSubnetPrefix string = '10.60.0.0/23'

@description('Delegated PostgreSQL subnet.')
param postgresSubnetPrefix string = '10.60.4.0/24'

@description('Private-endpoint subnet.')
param privateEndpointSubnetPrefix string = '10.60.5.0/24'

@description('Delegated Function App integration subnet.')
param functionsSubnetPrefix string = '10.60.6.0/24'

// ------------------------------------------------------------------ observability
@description('Log Analytics retention in days.')
@minValue(30)
@maxValue(730)
param logRetentionDays int = 30

@description('Daily Log Analytics ingestion cap in GB. -1 means uncapped.')
param logDailyQuotaGb int = -1

@description('Create diagnostic settings for the Container Apps environment and the Function App.')
param enableDiagnostics bool = true

// ------------------------------------------------------------------------ storage
@description('Storage redundancy.')
@allowed([
  'Standard_LRS'
  'Standard_ZRS'
  'Standard_GRS'
])
param storageSkuName string = 'Standard_LRS'

@description('Quota of the Qdrant Azure Files share, in GiB.')
@minValue(1)
param qdrantShareQuotaGb int = 100

// ----------------------------------------------------------------------- key vault
@description('Key Vault SKU.')
@allowed([
  'standard'
  'premium'
])
param keyVaultSkuName string = 'standard'

@description('Enable Key Vault purge protection. Irreversible; on for production.')
param keyVaultPurgeProtection bool = false

@description('Object id of the deploying principal, granted Key Vault Secrets Officer.')
param deployerPrincipalId string = ''

@description('Principal type of deployerPrincipalId.')
@allowed([
  'User'
  'Group'
  'ServicePrincipal'
])
param deployerPrincipalType string = 'User'

// ------------------------------------------------------------------------ postgres
@description('PostgreSQL compute SKU.')
param postgresSkuName string = 'Standard_B2s'

@description('PostgreSQL SKU tier matching postgresSkuName.')
@allowed([
  'Burstable'
  'GeneralPurpose'
  'MemoryOptimized'
])
param postgresSkuTier string = 'Burstable'

@description('PostgreSQL major version.')
@allowed([
  '15'
  '16'
  '17'
])
param postgresVersion string = '16'

@description('PostgreSQL data disk size in GiB.')
@allowed([
  32
  64
  128
  256
  512
])
param postgresStorageSizeGB int = 32

@description('PostgreSQL administrator login (password auth).')
param postgresAdminLogin string = 'ragadmin'

@description('Application database name; must equal RAG_POSTGRES_DB.')
param postgresDatabaseName string = 'rag'

@description('Point-in-time backup retention in days.')
@minValue(7)
@maxValue(35)
param postgresBackupRetentionDays int = 7

@description('Store PostgreSQL backups in the paired region as well.')
param postgresGeoRedundantBackup bool = false

@description('Zone-redundant PostgreSQL high availability. Needs GeneralPurpose or higher.')
param postgresZoneRedundantHa bool = false

@description('Entra object id of the PostgreSQL directory administrator. Empty skips it.')
param postgresEntraAdminObjectId string = ''

@description('Display name / UPN of the PostgreSQL Entra administrator.')
param postgresEntraAdminPrincipalName string = ''

@description('Principal type of the PostgreSQL Entra administrator.')
@allowed([
  'User'
  'Group'
  'ServicePrincipal'
])
param postgresEntraAdminPrincipalType string = 'User'

// ------------------------------------------------------------------- container apps
@description('Spread Container Apps replicas across availability zones.')
param containerAppsZoneRedundant bool = false

@description('API CPU cores, as a string.')
param apiCpu string = '1.0'

@description('API memory, e.g. 2Gi.')
param apiMemory string = '2Gi'

@description('Minimum API replicas. At least 1 keeps FastEmbed models warm.')
@minValue(1)
param apiMinReplicas int = 1

@description('Maximum API replicas.')
@minValue(1)
param apiMaxReplicas int = 5

@description('Concurrent requests per API replica before scaling out.')
param apiConcurrentRequests int = 20

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

@description('Browser origins the API accepts, as a JSON array string. Empty derives the web origin.')
param apiCorsOrigins string = ''

// -------------------------------------------------------------------------- qdrant
@description('Qdrant CPU cores, as a string.')
param qdrantCpu string = '1.0'

@description('Qdrant memory, e.g. 2Gi.')
param qdrantMemory string = '2Gi'

@description('Qdrant optimizer threads. Keep below the CPU allocation.')
@minValue(1)
param qdrantMaxOptimizationThreads int = 2

// ------------------------------------------------------------------------ functions
@description('Memory per Function App instance, in MB.')
@allowed([
  512
  2048
  4096
])
param functionsInstanceMemoryMB int = 2048

@description('Maximum concurrent Function App instances.')
@minValue(40)
@maxValue(1000)
param functionsMaximumInstanceCount int = 40

@description('Durable Functions task hub name.')
param durableTaskHubName string = 'ragingest'

// ------------------------------------------------------------------ ingest schedule
@description('Master switch for scheduled ingestion.')
param ingestEnabled bool = true

@description('Six-field NCRONTAB delta-refresh schedule. Default 02:30 daily.')
param ingestCron string = '0 30 2 * * *'

@description('IANA timezone for the schedule and the working-hours guard.')
param ingestTimezone string = 'UTC'

@description('First hour of the working day.')
@minValue(0)
@maxValue(23)
param ingestWorkingHoursStart int = 8

@description('End hour of the working day, exclusive.')
@minValue(0)
@maxValue(24)
param ingestWorkingHoursEnd int = 18

@description('Documents processed concurrently within one ingestion run.')
@minValue(1)
param ingestMaxParallelDocs int = 8

@description('Documents per Durable Functions activity batch.')
@minValue(1)
param ingestBatchSize int = 32

// --------------------------------------------------------------------------- names
var resourceToken = toLower(uniqueString(subscription().id, resourceGroup().id, environmentName))
var namePrefix = '${baseName}-${environmentName}'

var names = {
  logAnalytics: 'log-${namePrefix}-${resourceToken}'
  appInsights: 'appi-${namePrefix}-${resourceToken}'
  keyVault: take('kv-${baseName}${environmentName}${resourceToken}', 24)
  storage: take(toLower('st${baseName}${environmentName}${resourceToken}'), 24)
  virtualNetwork: 'vnet-${namePrefix}'
  postgres: 'psql-${namePrefix}-${resourceToken}'
  containerAppsEnvironment: 'cae-${namePrefix}-${resourceToken}'
  apiApp: 'ca-${namePrefix}-api'
  webApp: 'ca-${namePrefix}-web'
  qdrantApp: 'ca-${namePrefix}-qdrant'
  functionApp: 'func-${namePrefix}-${resourceToken}'
  functionPlan: 'plan-${namePrefix}-${resourceToken}'
  apiIdentity: 'id-${namePrefix}-api'
  webIdentity: 'id-${namePrefix}-web'
  ingestionIdentity: 'id-${namePrefix}-ingest'
  qdrantIdentity: 'id-${namePrefix}-qdrant'
}

var langfuseEnabled = !empty(langfusePublicKey) && !empty(langfuseSecretKey)
var piiHashSecretConfigured = !empty(piiHashSecret)

var commonTags = union(
  {
    'azd-env-name': environmentName
    application: 'productionizing-rag'
    environment: environmentName
  },
  tags
)

// ------------------------------------------------------------------------- modules
module observability 'modules/observability.bicep' = {
  name: 'observability'
  params: {
    logAnalyticsWorkspaceName: names.logAnalytics
    appInsightsName: names.appInsights
    location: location
    tags: commonTags
    logRetentionDays: logRetentionDays
    dailyQuotaGb: logDailyQuotaGb
    deployCore: true
  }
}

module network 'modules/network.bicep' = {
  name: 'network'
  params: {
    virtualNetworkName: names.virtualNetwork
    location: location
    tags: commonTags
    vnetAddressPrefix: vnetAddressPrefix
    containerAppsSubnetPrefix: containerAppsSubnetPrefix
    postgresSubnetPrefix: postgresSubnetPrefix
    privateEndpointSubnetPrefix: privateEndpointSubnetPrefix
    functionsSubnetPrefix: functionsSubnetPrefix
    postgresServerName: names.postgres
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    storageAccountName: names.storage
    location: location
    tags: commonTags
    skuName: storageSkuName
    qdrantShareQuotaGb: qdrantShareQuotaGb
  }
}

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    keyVaultName: names.keyVault
    location: location
    tags: commonTags
    skuName: keyVaultSkuName
    enablePurgeProtection: keyVaultPurgeProtection
    anthropicApiKey: anthropicApiKey
    qdrantApiKey: qdrantApiKey
    postgresAdminPassword: postgresAdminPassword
    langfusePublicKey: langfusePublicKey
    langfuseSecretKey: langfuseSecretKey
    piiHashSecret: piiHashSecret
  }
}

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    location: location
    tags: commonTags
    apiIdentityName: names.apiIdentity
    webIdentityName: names.webIdentity
    ingestionIdentityName: names.ingestionIdentity
    qdrantIdentityName: names.qdrantIdentity
    keyVaultName: keyVault.outputs.keyVaultName
    storageAccountName: storage.outputs.storageAccountName
    sourcesContainerName: storage.outputs.sourcesContainerName
    containerRegistryName: containerRegistryName
    deployerPrincipalId: deployerPrincipalId
    deployerPrincipalType: deployerPrincipalType
  }
}

module postgres 'modules/postgres.bicep' = {
  name: 'postgres'
  params: {
    serverName: names.postgres
    location: location
    tags: commonTags
    skuName: postgresSkuName
    skuTier: postgresSkuTier
    postgresVersion: postgresVersion
    storageSizeGB: postgresStorageSizeGB
    administratorLogin: postgresAdminLogin
    administratorPassword: postgresAdminPassword
    databaseName: postgresDatabaseName
    delegatedSubnetId: network.outputs.postgresSubnetId
    privateDnsZoneId: network.outputs.postgresPrivateDnsZoneId
    backupRetentionDays: postgresBackupRetentionDays
    geoRedundantBackup: postgresGeoRedundantBackup
    zoneRedundantHighAvailability: postgresZoneRedundantHa
    entraAdminObjectId: postgresEntraAdminObjectId
    entraAdminPrincipalName: postgresEntraAdminPrincipalName
    entraAdminPrincipalType: postgresEntraAdminPrincipalType
  }
}

module containerApps 'modules/containerapps.bicep' = {
  name: 'containerapps'
  params: {
    environmentName: names.containerAppsEnvironment
    location: location
    tags: commonTags
    logAnalyticsWorkspaceName: observability.outputs.logAnalyticsWorkspaceName
    infrastructureSubnetId: network.outputs.containerAppsSubnetId
    zoneRedundant: containerAppsZoneRedundant
    apiAppName: names.apiApp
    webAppName: names.webApp
    qdrantAppName: names.qdrantApp
    apiImage: apiImage
    webImage: webImage
    containerRegistryLoginServer: containerRegistryLoginServer
    apiIdentityId: identity.outputs.apiIdentityId
    apiIdentityClientId: identity.outputs.apiIdentityClientId
    webIdentityId: identity.outputs.webIdentityId
    apiCpu: apiCpu
    apiMemory: apiMemory
    apiMinReplicas: apiMinReplicas
    apiMaxReplicas: apiMaxReplicas
    apiConcurrentRequests: apiConcurrentRequests
    webCpu: webCpu
    webMemory: webMemory
    webMinReplicas: webMinReplicas
    webMaxReplicas: webMaxReplicas
    ragEnv: ragEnv
    logLevel: logLevel
    postgresHost: postgres.outputs.fullyQualifiedDomainName
    postgresUser: postgres.outputs.administratorLogin
    postgresDatabase: postgres.outputs.databaseName
    blobAccountUrl: storage.outputs.blobEndpoint
    sourcesContainerName: storage.outputs.sourcesContainerName
    rawContainerName: storage.outputs.rawContainerName
    manifestsContainerName: storage.outputs.manifestsContainerName
    ingestQueueName: storage.outputs.ingestQueueName
    keyVaultUri: keyVault.outputs.keyVaultUri
    secretUris: keyVault.outputs.secretUris
    entraTenantId: entraTenantId
    entraClientId: entraClientId
    entraAudience: entraAudience
    langfuseEnabled: langfuseEnabled
    langfuseHost: langfuseHost
    piiHashSecretConfigured: piiHashSecretConfigured
    ingestCron: ingestCron
    ingestTimezone: ingestTimezone
    ingestWorkingHoursStart: ingestWorkingHoursStart
    ingestWorkingHoursEnd: ingestWorkingHoursEnd
    apiCorsOrigins: apiCorsOrigins
  }
}

module qdrant 'modules/qdrant.bicep' = {
  name: 'qdrant'
  params: {
    appName: names.qdrantApp
    location: location
    tags: commonTags
    environmentId: containerApps.outputs.environmentId
    environmentName: containerApps.outputs.environmentName
    qdrantImage: qdrantImage
    storageAccountName: storage.outputs.storageAccountName
    fileShareName: storage.outputs.qdrantShareName
    qdrantApiKeySecretUri: keyVault.outputs.secretUris.qdrantApiKey
    qdrantIdentityId: identity.outputs.qdrantIdentityId
    cpu: qdrantCpu
    memory: qdrantMemory
    maxOptimizationThreads: qdrantMaxOptimizationThreads
    // Qdrant uses WARN/ERROR where Python uses WARNING/CRITICAL. Every branch is a
    // literal so the mapped value still satisfies the module's allowed-value list.
    logLevel: logLevel == 'DEBUG'
      ? 'DEBUG'
      : (logLevel == 'WARNING' ? 'WARN' : (logLevel == 'ERROR' || logLevel == 'CRITICAL' ? 'ERROR' : 'INFO'))
  }
}

module functions 'modules/functions.bicep' = {
  name: 'functions'
  params: {
    functionAppName: names.functionApp
    planName: names.functionPlan
    location: location
    tags: commonTags
    instanceMemoryMB: functionsInstanceMemoryMB
    maximumInstanceCount: functionsMaximumInstanceCount
    storageAccountName: storage.outputs.storageAccountName
    deploymentContainerName: storage.outputs.functionReleasesContainerName
    virtualNetworkSubnetId: network.outputs.functionsSubnetId
    appInsightsName: observability.outputs.appInsightsName
    ingestionIdentityId: identity.outputs.ingestionIdentityId
    ingestionIdentityClientId: identity.outputs.ingestionIdentityClientId
    durableTaskHubName: durableTaskHubName
    ragEnv: ragEnv
    logLevel: logLevel
    qdrantUrl: qdrant.outputs.internalUrl
    postgresHost: postgres.outputs.fullyQualifiedDomainName
    postgresUser: postgres.outputs.administratorLogin
    postgresDatabase: postgres.outputs.databaseName
    blobAccountUrl: storage.outputs.blobEndpoint
    sourcesContainerName: storage.outputs.sourcesContainerName
    rawContainerName: storage.outputs.rawContainerName
    manifestsContainerName: storage.outputs.manifestsContainerName
    ingestQueueName: storage.outputs.ingestQueueName
    keyVaultUri: keyVault.outputs.keyVaultUri
    secretUris: keyVault.outputs.secretUris
    entraTenantId: entraTenantId
    entraClientId: entraClientId
    langfuseEnabled: langfuseEnabled
    langfuseHost: langfuseHost
    piiHashSecretConfigured: piiHashSecretConfigured
    ingestEnabled: ingestEnabled
    ingestCron: ingestCron
    ingestTimezone: ingestTimezone
    ingestWorkingHoursStart: ingestWorkingHoursStart
    ingestWorkingHoursEnd: ingestWorkingHoursEnd
    ingestMaxParallelDocs: ingestMaxParallelDocs
    ingestBatchSize: ingestBatchSize
  }
}

// Second observability pass: the diagnostic targets only exist now.
module diagnostics 'modules/observability.bicep' = if (enableDiagnostics) {
  name: 'observability-diagnostics'
  params: {
    logAnalyticsWorkspaceName: observability.outputs.logAnalyticsWorkspaceName
    appInsightsName: observability.outputs.appInsightsName
    location: location
    tags: commonTags
    deployCore: false
    containerAppsEnvironmentName: containerApps.outputs.environmentName
    functionAppName: functions.outputs.functionAppName
  }
}

// ------------------------------------------------------------------------- outputs
@description('Public API base URL.')
output apiUrl string = containerApps.outputs.apiUrl

@description('Public API FQDN.')
output apiFqdn string = containerApps.outputs.apiFqdn

@description('Public web app URL.')
output webUrl string = containerApps.outputs.webUrl

@description('Internal Qdrant URL (RAG_QDRANT_URL for the workloads).')
output qdrantUrl string = qdrant.outputs.internalUrl

@description('Container Apps environment name.')
output containerAppsEnvironmentName string = containerApps.outputs.environmentName

@description('Container Apps environment default domain.')
output containerAppsDefaultDomain string = containerApps.outputs.defaultDomain

@description('API container app name, for az containerapp update --image.')
output apiAppName string = containerApps.outputs.apiAppName

@description('Web container app name.')
output webAppName string = containerApps.outputs.webAppName

@description('Qdrant container app name.')
output qdrantAppName string = qdrant.outputs.appName

@description('Key Vault name.')
output keyVaultName string = keyVault.outputs.keyVaultName

@description('Key Vault URI (RAG_AZURE_KEY_VAULT_URL).')
output keyVaultUri string = keyVault.outputs.keyVaultUri

@description('Key Vault secret names, so scripts never hard-code them.')
output keyVaultSecretNames object = keyVault.outputs.secretNames

@description('Storage account name.')
output storageAccountName string = storage.outputs.storageAccountName

@description('Blob endpoint (RAG_AZURE_BLOB_ACCOUNT_URL).')
output blobEndpoint string = storage.outputs.blobEndpoint

@description('Container holding source documents (RAG_AZURE_BLOB_CONTAINER).')
output sourcesContainerName string = storage.outputs.sourcesContainerName

@description('Container holding archived raw bytes (RAG_AZURE_BLOB_RAW_CONTAINER).')
output rawContainerName string = storage.outputs.rawContainerName

@description('Container holding ingest manifests (RAG_INGEST_MANIFEST_CONTAINER).')
output manifestsContainerName string = storage.outputs.manifestsContainerName

@description('Container holding Function App deployment packages.')
output functionReleasesContainerName string = storage.outputs.functionReleasesContainerName

@description('Queue driving queue-triggered ingestion (RAG_AZURE_STORAGE_QUEUE_NAME).')
output ingestQueueName string = storage.outputs.ingestQueueName

@description('PostgreSQL FQDN (RAG_POSTGRES_HOST).')
output postgresHost string = postgres.outputs.fullyQualifiedDomainName

@description('PostgreSQL database name (RAG_POSTGRES_DB).')
output postgresDatabase string = postgres.outputs.databaseName

@description('PostgreSQL administrator login (RAG_POSTGRES_USER).')
output postgresUser string = postgres.outputs.administratorLogin

@description('Function App name, for func azure functionapp publish.')
output functionAppName string = functions.outputs.functionAppName

@description('Function App default hostname.')
output functionAppHostName string = functions.outputs.defaultHostName

@description('Durable Functions task hub name.')
output durableTaskHubName string = functions.outputs.durableTaskHubName

@description('API managed identity client id (RAG_AZURE_CLIENT_ID for the API).')
output apiIdentityClientId string = identity.outputs.apiIdentityClientId

@description('Ingestion managed identity client id (RAG_AZURE_CLIENT_ID for the Function App).')
output ingestionIdentityClientId string = identity.outputs.ingestionIdentityClientId

@description('Log Analytics workspace name.')
output logAnalyticsWorkspaceName string = observability.outputs.logAnalyticsWorkspaceName

@description('Application Insights component name.')
output appInsightsName string = observability.outputs.appInsightsName

@description('Virtual network resource id.')
output virtualNetworkId string = network.outputs.virtualNetworkId

@description('Whether Langfuse tracing was enabled by this deployment.')
output langfuseEnabled bool = langfuseEnabled

@description('Deterministic resource-name token for this environment.')
output resourceToken string = resourceToken
