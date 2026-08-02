// ---------------------------------------------------------------------------------
// storage.bicep — one storage account holding everything the platform persists
// outside PostgreSQL and Qdrant.
//
// Blob containers:
//   sources        — documents the blob/local connectors ingest (RAG_AZURE_BLOB_CONTAINER)
//   raw            — archived raw bytes per ingested document (RAG_AZURE_BLOB_RAW_CONTAINER)
//   manifests      — per-(tenant, source) IngestManifest documents
//   function-releases — Flex Consumption deployment package container
//   eval-reports   — evaluation run artefacts
//
// Queue `rag-ingest` (+ poison) drives the queue-triggered ingestion function.
// File share `qdrant-storage` is mounted at /qdrant/storage by the Qdrant Container App.
//
// Shared-key access stays enabled because Container Apps Azure Files volumes
// authenticate with an account key — there is no managed-identity option for that
// mount. Every *application* data path uses managed identity (defaultToOAuth is on).
// ---------------------------------------------------------------------------------

@description('Storage account name. 3-24 lowercase alphanumeric characters.')
@minLength(3)
@maxLength(24)
param storageAccountName string

@description('Azure region.')
param location string

@description('Tags applied to every resource in this module.')
param tags object = {}

@description('Storage redundancy.')
@allowed([
  'Standard_LRS'
  'Standard_ZRS'
  'Standard_GRS'
])
param skuName string = 'Standard_LRS'

@description('Container holding source documents for the blob connector.')
param sourcesContainerName string = 'rag-documents'

@description('Container where ingestion archives raw fetched bytes.')
param rawContainerName string = 'rag-raw'

@description('Container holding per-source ingest manifests.')
param manifestsContainerName string = 'rag-manifests'

@description('Container holding Flex Consumption deployment packages.')
param functionReleasesContainerName string = 'function-releases'

@description('Container holding evaluation run reports.')
param evalReportsContainerName string = 'rag-eval-reports'

@description('Queue driving the queue-triggered ingestion function.')
param ingestQueueName string = 'rag-ingest'

@description('File share mounted at /qdrant/storage by the Qdrant Container App.')
param qdrantShareName string = 'qdrant-storage'

@description('Quota of the Qdrant file share, in GiB.')
@minValue(1)
param qdrantShareQuotaGb int = 100

@description('Blob soft-delete retention in days.')
@minValue(1)
@maxValue(365)
param blobRetentionDays int = 7

var blobContainerNames = [
  sourcesContainerName
  rawContainerName
  manifestsContainerName
  functionReleasesContainerName
  evalReportsContainerName
]

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: skuName
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    // Required by the Container Apps Azure Files volume used for Qdrant persistence.
    allowSharedKeyAccess: true
    defaultToOAuthAuthentication: true
    publicNetworkAccess: 'Enabled'
    allowCrossTenantReplication: false
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
    encryption: {
      keySource: 'Microsoft.Storage'
      requireInfrastructureEncryption: false
      services: {
        blob: {
          enabled: true
          keyType: 'Account'
        }
        file: {
          enabled: true
          keyType: 'Account'
        }
        queue: {
          enabled: true
          keyType: 'Account'
        }
        table: {
          enabled: true
          keyType: 'Account'
        }
      }
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: blobRetentionDays
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: blobRetentionDays
    }
    isVersioningEnabled: false
  }
}

resource blobContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [
  for name in blobContainerNames: {
    parent: blobService
    name: name
    properties: {
      publicAccess: 'None'
    }
  }
]

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    shareDeleteRetentionPolicy: {
      enabled: true
      days: blobRetentionDays
    }
  }
}

resource qdrantShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: qdrantShareName
  properties: {
    shareQuota: qdrantShareQuotaGb
    enabledProtocols: 'SMB'
    accessTier: 'TransactionOptimized'
  }
}

resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {}
}

resource ingestQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2023-05-01' = {
  parent: queueService
  name: ingestQueueName
  properties: {}
}

resource ingestPoisonQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2023-05-01' = {
  parent: queueService
  name: '${ingestQueueName}-poison'
  properties: {}
}

// Durable Functions creates its own history/instance tables at runtime; the table
// service must exist and the ingestion identity needs Table Data Contributor.
resource tableService 'Microsoft.Storage/storageAccounts/tableServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {}
}

@description('Storage account name.')
output storageAccountName string = storageAccount.name

@description('Storage account resource id.')
output storageAccountId string = storageAccount.id

@description('Blob service endpoint, e.g. https://stragdev….blob.core.windows.net/.')
output blobEndpoint string = storageAccount.properties.primaryEndpoints.blob

@description('Queue service endpoint.')
output queueEndpoint string = storageAccount.properties.primaryEndpoints.queue

@description('Table service endpoint.')
output tableEndpoint string = storageAccount.properties.primaryEndpoints.table

@description('Container holding source documents.')
output sourcesContainerName string = sourcesContainerName

@description('Container holding archived raw bytes.')
output rawContainerName string = rawContainerName

@description('Container holding ingest manifests.')
output manifestsContainerName string = manifestsContainerName

@description('Container holding Flex Consumption deployment packages.')
output functionReleasesContainerName string = functionReleasesContainerName

@description('Container holding evaluation reports.')
output evalReportsContainerName string = evalReportsContainerName

@description('Queue driving queue-triggered ingestion.')
output ingestQueueName string = ingestQueue.name

@description('File share mounted for Qdrant persistence.')
output qdrantShareName string = qdrantShare.name
