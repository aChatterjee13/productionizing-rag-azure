// ---------------------------------------------------------------------------------
// prod.bicepparam — production environment.
//
// Differences that matter versus dev:
//   * ragEnv = 'production', which makes ragcore refuse entra_dev_mode and
//     tool_allow_insecure_http at startup.
//   * Key Vault purge protection on (irreversible, deliberately).
//   * PostgreSQL GeneralPurpose + zone-redundant HA + 35-day geo-redundant backups.
//   * Zone-redundant Container Apps environment, two API replicas minimum.
//   * Larger Qdrant allocation and a 256 GiB persistent share.
//
// No secret values: everything sensitive is read from the environment by deploy.sh.
// ---------------------------------------------------------------------------------

using '../main.bicep'

param baseName = readEnvironmentVariable('RAG_BASE_NAME', 'rag')
param environmentName = 'prod'
param location = readEnvironmentVariable('AZURE_LOCATION', 'westeurope')
param ragEnv = 'production'
param logLevel = readEnvironmentVariable('RAG_LOG_LEVEL', 'INFO')

param tags = {
  environment: 'prod'
  costCenter: readEnvironmentVariable('AZURE_COST_CENTER', 'engineering')
  dataClassification: 'confidential'
}

// ------------------------------------------------------------------------- secrets
param anthropicApiKey = readEnvironmentVariable('ANTHROPIC_API_KEY', '')
param qdrantApiKey = readEnvironmentVariable('QDRANT_API_KEY', '')
param postgresAdminPassword = readEnvironmentVariable('POSTGRES_ADMIN_PASSWORD', '')
param langfusePublicKey = readEnvironmentVariable('LANGFUSE_PUBLIC_KEY', '')
param langfuseSecretKey = readEnvironmentVariable('LANGFUSE_SECRET_KEY', '')
param piiHashSecret = readEnvironmentVariable('PII_HASH_SECRET', '')
param langfuseHost = readEnvironmentVariable('LANGFUSE_HOST', '')

// --------------------------------------------------------------------------- entra
param entraTenantId = readEnvironmentVariable('ENTRA_TENANT_ID', '')
param entraClientId = readEnvironmentVariable('ENTRA_CLIENT_ID', '')
param entraAudience = readEnvironmentVariable('ENTRA_AUDIENCE', '')

// -------------------------------------------------------------------------- images
param apiImage = readEnvironmentVariable('API_IMAGE', '')
param webImage = readEnvironmentVariable('WEB_IMAGE', '')
param qdrantImage = readEnvironmentVariable('QDRANT_IMAGE', 'qdrant/qdrant:v1.12.4')
param containerRegistryLoginServer = readEnvironmentVariable('ACR_LOGIN_SERVER', '')
param containerRegistryName = readEnvironmentVariable('ACR_NAME', '')

// ---------------------------------------------------------------------- networking
param vnetAddressPrefix = '10.70.0.0/16'
param containerAppsSubnetPrefix = '10.70.0.0/23'
param postgresSubnetPrefix = '10.70.4.0/24'
param privateEndpointSubnetPrefix = '10.70.5.0/24'
param functionsSubnetPrefix = '10.70.6.0/24'

// -------------------------------------------------------------------- observability
param logRetentionDays = 90
param logDailyQuotaGb = -1
param enableDiagnostics = true

// -------------------------------------------------------------- storage / key vault
param storageSkuName = 'Standard_ZRS'
param qdrantShareQuotaGb = 256
param keyVaultSkuName = 'standard'
param keyVaultPurgeProtection = true
param deployerPrincipalId = readEnvironmentVariable('DEPLOYER_PRINCIPAL_ID', '')
param deployerPrincipalType = readEnvironmentVariable('DEPLOYER_PRINCIPAL_TYPE', 'ServicePrincipal')

// ------------------------------------------------------------------------ postgres
param postgresSkuName = 'Standard_D2ds_v5'
param postgresSkuTier = 'GeneralPurpose'
param postgresVersion = '16'
param postgresStorageSizeGB = 128
param postgresBackupRetentionDays = 35
param postgresGeoRedundantBackup = true
param postgresZoneRedundantHa = true
param postgresEntraAdminObjectId = readEnvironmentVariable('POSTGRES_ENTRA_ADMIN_OBJECT_ID', '')
param postgresEntraAdminPrincipalName = readEnvironmentVariable('POSTGRES_ENTRA_ADMIN_PRINCIPAL_NAME', '')
param postgresEntraAdminPrincipalType = readEnvironmentVariable('POSTGRES_ENTRA_ADMIN_PRINCIPAL_TYPE', 'Group')

// ------------------------------------------------------------------ container apps
param containerAppsZoneRedundant = true
param apiCpu = '2.0'
param apiMemory = '4Gi'
param apiMinReplicas = 2
param apiMaxReplicas = 10
param apiConcurrentRequests = 20
param webCpu = '0.5'
param webMemory = '1Gi'
param webMinReplicas = 2
param webMaxReplicas = 5

// -------------------------------------------------------------------------- qdrant
// Still exactly one replica: a single Azure Files volume backs /qdrant/storage, and a
// second replica mounting it would corrupt the write-ahead log.
param qdrantCpu = '2.0'
param qdrantMemory = '4Gi'
param qdrantMaxOptimizationThreads = 2

// ----------------------------------------------------------------------- functions
param functionsInstanceMemoryMB = 4096
param functionsMaximumInstanceCount = 100
param durableTaskHubName = 'ragingestprod'

// ------------------------------------------------------------------ ingest schedule
param ingestEnabled = true
param ingestCron = readEnvironmentVariable('RAG_INGEST_CRON', '0 30 2 * * *')
param ingestTimezone = readEnvironmentVariable('RAG_INGEST_TIMEZONE', 'Europe/Berlin')
param ingestWorkingHoursStart = 7
param ingestWorkingHoursEnd = 20
param ingestMaxParallelDocs = 8
param ingestBatchSize = 32
