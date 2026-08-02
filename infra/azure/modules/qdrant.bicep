// ---------------------------------------------------------------------------------
// qdrant.bicep — Qdrant as an internal-only Container App with durable storage.
//
// Non-negotiables encoded here:
//   * Persistence. /qdrant/storage is an Azure Files volume. An ephemeral Qdrant would
//     silently lose every collection on the next revision, so the volume is mandatory
//     and the file share is created by storage.bicep.
//   * minReplicas >= 1. A vector database must not scale to zero: cold-starting it
//     would drop the HNSW index from memory and stall every retrieval.
//   * maxReplicas defaults to 1. Two replicas would mount the same file share and
//     corrupt the write-ahead log; horizontal scale means a real Qdrant cluster with
//     one volume per node, which is out of scope for this template.
//   * Internal ingress only. Qdrant is never reachable from the internet; the API
//     talks to it over the environment's internal DNS name.
//   * The API key arrives as a Key Vault reference resolved by a managed identity.
//     The storage account key needed by the Azure Files mount is read inside this
//     module with listKeys() and is never a parameter or an output.
// ---------------------------------------------------------------------------------

@description('Qdrant container app name.')
param appName string

@description('Azure region.')
param location string

@description('Tags applied to every resource in this module.')
param tags object = {}

@description('Container Apps managed environment resource id.')
param environmentId string

@description('Container Apps managed environment name; the storage link is a child of it.')
param environmentName string

@description('Pinned Qdrant image. Never use a floating tag for a stateful service.')
param qdrantImage string = 'qdrant/qdrant:v1.12.4'

@description('Storage account holding the Azure Files share.')
param storageAccountName string

@description('File share mounted at /qdrant/storage.')
param fileShareName string

@description('Name of the managed-environment storage link.')
param environmentStorageName string = 'qdrant-storage'

@description('Versionless Key Vault secret URI of the Qdrant API key.')
param qdrantApiKeySecretUri string

@description('Resource id of the user-assigned identity that may read the Key Vault secret.')
param qdrantIdentityId string

@description('REST port Qdrant listens on.')
param httpPort int = 6333

@description('gRPC port Qdrant listens on inside the container. Not published through ingress: Container Apps HTTP ingress exposes a single port, and ragcore defaults to REST (qdrant_prefer_grpc=false).')
param grpcPort int = 6334

@description('CPU cores, as a string so it can be passed to json().')
param cpu string = '1.0'

@description('Memory, e.g. 2Gi. Container Apps requires memory = 2x CPU in GiB.')
param memory string = '2Gi'

@description('Minimum replicas. Must stay at 1: a vector DB must not scale to zero.')
@minValue(1)
@maxValue(1)
param minReplicas int = 1

@description('Maximum replicas. Must stay at 1 while a single Azure Files volume backs storage.')
@minValue(1)
@maxValue(1)
param maxReplicas int = 1

@description('Qdrant log level.')
@allowed([
  'TRACE'
  'DEBUG'
  'INFO'
  'WARN'
  'ERROR'
])
param logLevel string = 'INFO'

@description('Optimizer threads. Keep below the CPU allocation so search stays responsive.')
@minValue(1)
param maxOptimizationThreads int = 2

var volumeName = 'qdrant-storage-volume'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}

// Container Apps Azure Files volumes authenticate with an account key; there is no
// managed-identity option for the mount. The key is resolved here and never leaves
// this module.
resource environmentStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: managedEnvironment
  name: environmentStorageName
  properties: {
    azureFile: {
      accountName: storageAccountName
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: fileShareName
      accessMode: 'ReadWrite'
    }
  }
}

resource qdrantApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${qdrantIdentityId}': {}
    }
  }
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        // Internal only: reachable at <appName>.internal.<defaultDomain>.
        external: false
        targetPort: httpPort
        transport: 'auto'
        allowInsecure: true
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      secrets: [
        {
          name: 'qdrant-api-key'
          keyVaultUrl: qdrantApiKeySecretUri
          identity: qdrantIdentityId
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'qdrant'
          image: qdrantImage
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: [
            {
              name: 'QDRANT__SERVICE__HTTP_PORT'
              value: string(httpPort)
            }
            {
              name: 'QDRANT__SERVICE__GRPC_PORT'
              value: string(grpcPort)
            }
            {
              name: 'QDRANT__SERVICE__API_KEY'
              secretRef: 'qdrant-api-key'
            }
            {
              name: 'QDRANT__SERVICE__ENABLE_CORS'
              value: 'false'
            }
            {
              name: 'QDRANT__LOG_LEVEL'
              value: logLevel
            }
            {
              name: 'QDRANT__STORAGE__STORAGE_PATH'
              value: '/qdrant/storage'
            }
            {
              name: 'QDRANT__STORAGE__SNAPSHOTS_PATH'
              value: '/qdrant/storage/snapshots'
            }
            {
              name: 'QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS'
              value: string(maxOptimizationThreads)
            }
            {
              name: 'QDRANT__TELEMETRY_DISABLED'
              value: 'true'
            }
          ]
          volumeMounts: [
            {
              volumeName: volumeName
              mountPath: '/qdrant/storage'
            }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/readyz'
                port: httpPort
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/readyz'
                port: httpPort
                scheme: 'HTTP'
              }
              periodSeconds: 10
              timeoutSeconds: 3
              failureThreshold: 3
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/livez'
                port: httpPort
                scheme: 'HTTP'
              }
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 5
            }
          ]
        }
      ]
      volumes: [
        {
          name: volumeName
          storageType: 'AzureFile'
          storageName: environmentStorage.name
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
    }
  }
}

@description('Qdrant container app name.')
output appName string = qdrantApp.name

@description('Internal FQDN of the Qdrant app.')
output internalFqdn string = qdrantApp.properties.configuration.ingress.fqdn

@description('URL other apps use for RAG_QDRANT_URL.')
output internalUrl string = 'http://${qdrantApp.properties.configuration.ingress.fqdn}'

@description('Name of the managed-environment Azure Files storage link.')
output environmentStorageName string = environmentStorage.name
