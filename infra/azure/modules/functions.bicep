// ---------------------------------------------------------------------------------
// functions.bicep — Flex Consumption Function App (Python 3.13) running ingestion.
//
// * Flex Consumption (FC1): per-function scaling, VNet integration, no Always On cost.
//   The deployment package lives in a blob container and is authenticated with the
//   user-assigned identity — no storage connection string anywhere.
// * Identity: system-assigned **and** user-assigned. The user-assigned identity is the
//   one that already holds the Storage and Key Vault role assignments (see
//   identity.bicep), which is what lets the app start on its first deployment; the
//   system-assigned identity is enabled for anything that needs the app's own principal.
// * Durable Functions keeps its task hub in this storage account's blobs, queues and
//   tables, addressed with the identity-based AzureWebJobsStorage__* settings.
// * All RAG_* settings mirror the API so ragcore.settings resolves identically in both
//   processes; secrets are Key Vault references.
// ---------------------------------------------------------------------------------

@description('Function App name.')
param functionAppName string

@description('Flex Consumption plan name.')
param planName string

@description('Azure region.')
param location string

@description('Tags applied to every resource in this module.')
param tags object = {}

@description('Python runtime version for the Flex Consumption app.')
@allowed([
  '3.11'
  '3.12'
  '3.13'
])
param pythonVersion string = '3.13'

@description('Memory per instance, in MB.')
@allowed([
  512
  2048
  4096
])
param instanceMemoryMB int = 2048

@description('Maximum concurrent instances.')
@minValue(40)
@maxValue(1000)
param maximumInstanceCount int = 40

@description('Always-ready instance count for the HTTP trigger group. 0 keeps it scale-to-zero.')
@minValue(0)
param httpAlwaysReadyInstances int = 0

@description('Storage account backing deployment, Durable task hub state and ingest queues.')
param storageAccountName string

@description('Blob container holding the deployment package.')
param deploymentContainerName string

@description('Subnet used for VNet integration so the app can reach private PostgreSQL.')
param virtualNetworkSubnetId string

@description('Application Insights component name. Its connection string is read here, not passed in.')
param appInsightsName string

@description('Resource id of the ingestion user-assigned identity.')
param ingestionIdentityId string

@description('Client id of the ingestion user-assigned identity.')
param ingestionIdentityClientId string

@description('Durable Functions task hub name. Change it to isolate two deployments sharing a storage account.')
param durableTaskHubName string = 'ragingest'

// --------------------------------------------------------------- app configuration
@description('Value of RAG_ENV inside the Function App.')
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

@description('Qdrant URL (internal Container Apps address).')
param qdrantUrl string

@description('PostgreSQL host.')
param postgresHost string

@description('PostgreSQL user.')
param postgresUser string

@description('PostgreSQL database.')
param postgresDatabase string

@description('Blob service endpoint (RAG_AZURE_BLOB_ACCOUNT_URL).')
param blobAccountUrl string

@description('Container holding source documents.')
param sourcesContainerName string

@description('Container holding archived raw bytes.')
param rawContainerName string

@description('Container holding ingest manifests.')
param manifestsContainerName string

@description('Queue driving queue-triggered ingestion.')
param ingestQueueName string

@description('Key Vault URI.')
param keyVaultUri string

@description('Versionless Key Vault secret URIs from keyvault.bicep.')
param secretUris object

@description('Entra directory (tenant) id.')
param entraTenantId string

@description('Entra API application (client) id.')
param entraClientId string

@description('Enable Langfuse tracing from the ingestion process.')
param langfuseEnabled bool = false

@description('Langfuse base URL.')
param langfuseHost string = ''

@description('Store a Key Vault reference for the PII HMAC key.')
param piiHashSecretConfigured bool = false

// ------------------------------------------------------------------- ingest schedule
@description('Master switch for scheduled ingestion (RAG_INGEST_ENABLED).')
param ingestEnabled bool = true

@description('Six-field NCRONTAB schedule the timer trigger binds to (RAG_INGEST_CRON).')
param ingestCron string = '0 30 2 * * *'

@description('IANA timezone for the schedule and the working-hours guard.')
param ingestTimezone string = 'UTC'

@description('First hour of the working day.')
param ingestWorkingHoursStart int = 8

@description('End hour of the working day, exclusive.')
param ingestWorkingHoursEnd int = 18

@description('Documents processed concurrently within one run.')
param ingestMaxParallelDocs int = 8

@description('Documents per Durable Functions activity batch.')
param ingestBatchSize int = 32

@description('Extra app settings appended verbatim.')
param additionalAppSettings array = []

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: appInsightsName
}

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: planName
  location: location
  tags: tags
  kind: 'functionapp'
  sku: {
    tier: 'FlexConsumption'
    name: 'FC1'
  }
  properties: {
    reserved: true
  }
}

var deploymentStorageUrl = '${storageAccount.properties.primaryEndpoints.blob}${deploymentContainerName}'

var baseAppSettings = [
  {
    name: 'FUNCTIONS_EXTENSION_VERSION'
    value: '~4'
  }
  {
    name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
    value: appInsights.properties.ConnectionString
  }
  {
    // Identity-based host storage: no connection string. Requires Blob Data Owner,
    // Queue Data Contributor and Table Data Contributor on the account.
    name: 'AzureWebJobsStorage__accountName'
    value: storageAccountName
  }
  {
    name: 'AzureWebJobsStorage__credential'
    value: 'managedidentity'
  }
  {
    name: 'AzureWebJobsStorage__clientId'
    value: ingestionIdentityClientId
  }
  {
    name: 'AzureWebJobsStorage__blobServiceUri'
    value: storageAccount.properties.primaryEndpoints.blob
  }
  {
    name: 'AzureWebJobsStorage__queueServiceUri'
    value: storageAccount.properties.primaryEndpoints.queue
  }
  {
    name: 'AzureWebJobsStorage__tableServiceUri'
    value: storageAccount.properties.primaryEndpoints.table
  }
  {
    // Durable Functions task hub. Overrides host.json so two environments can share
    // a storage account without colliding.
    name: 'AzureFunctionsJobHost__extensions__durableTask__hubName'
    value: durableTaskHubName
  }
  {
    name: 'PYTHON_ENABLE_INIT_INDEXING'
    value: '1'
  }
  {
    name: 'PYTHON_ISOLATE_WORKER_DEPENDENCIES'
    value: '1'
  }
  // ------------------------------------------------------------------ ragcore config
  {
    name: 'RAG_ENV'
    value: ragEnv
  }
  {
    name: 'RAG_SERVICE_NAME'
    value: 'rag-ingestion'
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
    value: qdrantUrl
  }
  {
    name: 'RAG_QDRANT_API_KEY'
    value: '@Microsoft.KeyVault(SecretUri=${secretUris.qdrantApiKey})'
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
    value: '@Microsoft.KeyVault(SecretUri=${secretUris.postgresAdminPassword})'
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
    name: 'RAG_REDIS_ENABLED'
    value: 'false'
  }
  {
    name: 'RAG_ANTHROPIC_API_KEY'
    value: '@Microsoft.KeyVault(SecretUri=${secretUris.anthropicApiKey})'
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
    value: ingestionIdentityClientId
  }
  {
    name: 'RAG_INGEST_ENABLED'
    value: ingestEnabled ? 'true' : 'false'
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
    name: 'RAG_INGEST_MAX_PARALLEL_DOCS'
    value: string(ingestMaxParallelDocs)
  }
  {
    name: 'RAG_INGEST_BATCH_SIZE'
    value: string(ingestBatchSize)
  }
  {
    // The timer trigger binds to %RAG_INGEST_CRON%; WEBSITE_TIME_ZONE makes the host
    // evaluate NCRONTAB in the same zone as ragcore's working-hours guard.
    name: 'WEBSITE_TIME_ZONE'
    value: ingestTimezone
  }
  {
    name: 'RAG_EMBEDDING_CACHE_DIR'
    value: '/tmp/fastembed'
  }
  {
    name: 'RAG_LANGFUSE_ENABLED'
    value: langfuseEnabled ? 'true' : 'false'
  }
]

var langfuseSettings = langfuseEnabled
  ? [
      {
        name: 'RAG_LANGFUSE_HOST'
        value: langfuseHost
      }
      {
        name: 'RAG_LANGFUSE_PUBLIC_KEY'
        value: '@Microsoft.KeyVault(SecretUri=${secretUris.langfusePublicKey})'
      }
      {
        name: 'RAG_LANGFUSE_SECRET_KEY'
        value: '@Microsoft.KeyVault(SecretUri=${secretUris.langfuseSecretKey})'
      }
    ]
  : []

var piiSettings = piiHashSecretConfigured
  ? [
      {
        name: 'RAG_PII_HASH_SECRET'
        value: '@Microsoft.KeyVault(SecretUri=${secretUris.piiHashSecret})'
      }
    ]
  : []

resource functionApp 'Microsoft.Web/sites@2023-12-01' = {
  name: functionAppName
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned, UserAssigned'
    userAssignedIdentities: {
      '${ingestionIdentityId}': {}
    }
  }
  properties: {
    serverFarmId: plan.id
    httpsOnly: true
    virtualNetworkSubnetId: virtualNetworkSubnetId
    vnetRouteAllEnabled: true
    // Key Vault references resolve through the user-assigned identity, which already
    // holds Key Vault Secrets User.
    keyVaultReferenceIdentity: ingestionIdentityId
    functionAppConfig: {
      deployment: {
        storage: {
          type: 'blobContainer'
          value: deploymentStorageUrl
          authentication: {
            type: 'UserAssignedIdentity'
            userAssignedIdentityResourceId: ingestionIdentityId
          }
        }
      }
      runtime: {
        name: 'python'
        version: pythonVersion
      }
      scaleAndConcurrency: {
        instanceMemoryMB: instanceMemoryMB
        maximumInstanceCount: maximumInstanceCount
        alwaysReady: httpAlwaysReadyInstances > 0
          ? [
              {
                name: 'http'
                instanceCount: httpAlwaysReadyInstances
              }
            ]
          : []
      }
    }
    siteConfig: {
      minTlsVersion: '1.2'
      ftpsState: 'Disabled'
      appSettings: concat(baseAppSettings, langfuseSettings, piiSettings, additionalAppSettings)
    }
  }
}

@description('Function App name.')
output functionAppName string = functionApp.name

@description('Function App resource id, consumed by the diagnostics wiring.')
output functionAppId string = functionApp.id

@description('Default hostname of the Function App.')
output defaultHostName string = functionApp.properties.defaultHostName

@description('System-assigned principal id, for any grant made outside this template.')
output systemAssignedPrincipalId string = functionApp.identity.principalId

@description('Flex Consumption plan resource id.')
output planId string = plan.id

@description('Durable Functions task hub name.')
output durableTaskHubName string = durableTaskHubName
