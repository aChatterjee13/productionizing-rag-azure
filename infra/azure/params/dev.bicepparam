// ---------------------------------------------------------------------------------
// dev.bicepparam — development environment.
//
// Every secret and every environment-specific identifier is read from the process
// environment, so this file is safe to commit: it contains no secret values.
// `deploy.sh` exports the variables (generating or reusing Qdrant/PostgreSQL
// credentials from Key Vault) before invoking `az deployment group create`.
// ---------------------------------------------------------------------------------

using '../main.bicep'

param baseName = readEnvironmentVariable('RAG_BASE_NAME', 'rag')
param environmentName = 'dev'
param location = readEnvironmentVariable('AZURE_LOCATION', 'westeurope')
param ragEnv = 'dev'
param logLevel = readEnvironmentVariable('RAG_LOG_LEVEL', 'INFO')

param tags = {
  environment: 'dev'
  costCenter: readEnvironmentVariable('AZURE_COST_CENTER', 'engineering')
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
param vnetAddressPrefix = '10.60.0.0/16'
param containerAppsSubnetPrefix = '10.60.0.0/23'
param postgresSubnetPrefix = '10.60.4.0/24'
param privateEndpointSubnetPrefix = '10.60.5.0/24'
param functionsSubnetPrefix = '10.60.6.0/24'

// -------------------------------------------------------------------- observability
param logRetentionDays = 30
param logDailyQuotaGb = 1
param enableDiagnostics = true

// -------------------------------------------------------------- storage / key vault
param storageSkuName = 'Standard_LRS'
param qdrantShareQuotaGb = 32
param keyVaultSkuName = 'standard'
param keyVaultPurgeProtection = false
param deployerPrincipalId = readEnvironmentVariable('DEPLOYER_PRINCIPAL_ID', '')
param deployerPrincipalType = readEnvironmentVariable('DEPLOYER_PRINCIPAL_TYPE', 'User')

// ------------------------------------------------------------------------ postgres
param postgresSkuName = 'Standard_B2s'
param postgresSkuTier = 'Burstable'
param postgresVersion = '16'
param postgresStorageSizeGB = 32
param postgresBackupRetentionDays = 7
param postgresGeoRedundantBackup = false
param postgresZoneRedundantHa = false
param postgresEntraAdminObjectId = readEnvironmentVariable('POSTGRES_ENTRA_ADMIN_OBJECT_ID', '')
param postgresEntraAdminPrincipalName = readEnvironmentVariable('POSTGRES_ENTRA_ADMIN_PRINCIPAL_NAME', '')
param postgresEntraAdminPrincipalType = readEnvironmentVariable('POSTGRES_ENTRA_ADMIN_PRINCIPAL_TYPE', 'User')

// ------------------------------------------------------------------ container apps
param containerAppsZoneRedundant = false
param apiCpu = '1.0'
param apiMemory = '2Gi'
param apiMinReplicas = 1
param apiMaxReplicas = 2
param apiConcurrentRequests = 20
param webCpu = '0.25'
param webMemory = '0.5Gi'
param webMinReplicas = 1
param webMaxReplicas = 2

// -------------------------------------------------------------------------- qdrant
param qdrantCpu = '1.0'
param qdrantMemory = '2Gi'
param qdrantMaxOptimizationThreads = 1

// ----------------------------------------------------------------------- functions
param functionsInstanceMemoryMB = 2048
param functionsMaximumInstanceCount = 40
param durableTaskHubName = 'ragingestdev'

// ------------------------------------------------------------------ ingest schedule
// 02:30 daily, outside working hours. Six-field NCRONTAB as Azure Functions requires.
param ingestEnabled = true
param ingestCron = readEnvironmentVariable('RAG_INGEST_CRON', '0 30 2 * * *')
param ingestTimezone = readEnvironmentVariable('RAG_INGEST_TIMEZONE', 'UTC')
param ingestWorkingHoursStart = 8
param ingestWorkingHoursEnd = 18
param ingestMaxParallelDocs = 4
param ingestBatchSize = 16
