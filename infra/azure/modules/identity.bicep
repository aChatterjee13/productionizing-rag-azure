// ---------------------------------------------------------------------------------
// identity.bicep — user-assigned managed identities and every role assignment the
// workloads need. No workload in this platform holds a connection string it could
// have obtained with an identity instead.
//
// Identities
//   api        — FastAPI Container App: reads Key Vault secrets, reads/writes the
//                sources container (multipart upload), pulls its image from ACR.
//   web        — nginx Container App: pulls its image from ACR. Nothing else.
//   ingestion  — Function App data plane: Key Vault secrets, plus blob/queue/table
//                access for Durable Functions state, the Flex Consumption deployment
//                container, the source documents and the ingest manifests.
//   qdrant     — Qdrant Container App: resolves its API key from Key Vault.
//
// Why the Function App uses a user-assigned identity for data access: role
// assignments must exist before the app starts, but a system-assigned principal id
// only exists after the app is created. A user-assigned identity provisioned here
// removes that cycle. functions.bicep still enables the system-assigned identity
// alongside it for anything that requires the app's own principal.
// ---------------------------------------------------------------------------------

@description('Azure region for the managed identities.')
param location string

@description('Tags applied to every identity.')
param tags object = {}

@description('Name of the API user-assigned identity.')
param apiIdentityName string

@description('Name of the web user-assigned identity.')
param webIdentityName string

@description('Name of the ingestion (Function App) user-assigned identity.')
param ingestionIdentityName string

@description('Name of the Qdrant user-assigned identity.')
param qdrantIdentityName string

@description('Key Vault to grant Secrets User on.')
param keyVaultName string

@description('Storage account to grant data-plane roles on.')
param storageAccountName string

@description('Blob container holding source documents.')
param sourcesContainerName string

@description('Container registry in this resource group to grant AcrPull on. Empty skips it.')
param containerRegistryName string = ''

@description('Object id of the deploying principal, granted Key Vault Secrets Officer so deploy.sh can seed secrets. Empty skips it.')
param deployerPrincipalId string = ''

@description('Principal type of deployerPrincipalId.')
@allowed([
  'User'
  'Group'
  'ServicePrincipal'
])
param deployerPrincipalType string = 'User'

// Built-in role definition GUIDs.
var roleIds = {
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
  keyVaultSecretsOfficer: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  storageBlobDataReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  storageBlobDataOwner: 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
  storageQueueDataContributor: '974c5e8b-45b9-4653-ba55-5f855dd0fb88'
  storageTableDataContributor: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3'
}

resource apiIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: apiIdentityName
  location: location
  tags: tags
}

resource webIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: webIdentityName
  location: location
  tags: tags
}

resource ingestionIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: ingestionIdentityName
  location: location
  tags: tags
}

resource qdrantIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: qdrantIdentityName
  location: location
  tags: tags
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storageAccount
  name: 'default'
}

resource sourcesContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' existing = {
  parent: blobService
  name: sourcesContainerName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: empty(containerRegistryName) ? 'placeholdercontainerregistry' : containerRegistryName
}

// ------------------------------------------------------------------ Key Vault RBAC
// Role-assignment names must be computable before the deployment starts, so every guid()
// is seeded with the identity's *resource id* (compile-time) rather than its principal
// id (runtime). The principal id is only used inside properties, where that is allowed.
var roleDefinitionScope = 'Microsoft.Authorization/roleDefinitions'

resource apiKeyVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, apiIdentity.id, roleIds.keyVaultSecretsUser)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.keyVaultSecretsUser)
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource ingestionKeyVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, ingestionIdentity.id, roleIds.keyVaultSecretsUser)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.keyVaultSecretsUser)
    principalId: ingestionIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource qdrantKeyVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, qdrantIdentity.id, roleIds.keyVaultSecretsUser)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.keyVaultSecretsUser)
    principalId: qdrantIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource deployerSecretsOfficer 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerPrincipalId)) {
  scope: keyVault
  name: guid(keyVault.id, deployerPrincipalId, roleIds.keyVaultSecretsOfficer)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.keyVaultSecretsOfficer)
    principalId: deployerPrincipalId
    principalType: deployerPrincipalType
  }
}

// -------------------------------------------------------------------- Storage RBAC
// Ingestion needs the account scope: Durable Functions keeps its task hub in blobs,
// queues and tables, and Flex Consumption reads its deployment package from a blob
// container. Blob Data Owner subsumes the read-only access it needs to the sources
// container, so no narrower duplicate assignment is created for it.
var ingestionAccountRoles = [
  roleIds.storageBlobDataOwner
  roleIds.storageQueueDataContributor
  roleIds.storageTableDataContributor
]

resource ingestionStorageRoles 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for roleId in ingestionAccountRoles: {
    scope: storageAccount
    name: guid(storageAccount.id, ingestionIdentity.id, roleId)
    properties: {
      roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleId)
      principalId: ingestionIdentity.properties.principalId
      principalType: 'ServicePrincipal'
    }
  }
]

// The API only reads blobs account-wide (lineage, manifests, archived raw copies)…
resource apiStorageReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, apiIdentity.id, roleIds.storageBlobDataReader)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.storageBlobDataReader)
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// …and writes only into the sources container, for POST /documents uploads.
resource apiSourcesContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: sourcesContainer
  name: guid(sourcesContainer.id, apiIdentity.id, roleIds.storageBlobDataContributor)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.storageBlobDataContributor)
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ------------------------------------------------------------------------ ACR pull
resource apiAcrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(containerRegistryName)) {
  scope: containerRegistry
  name: guid(containerRegistry.id, apiIdentity.id, roleIds.acrPull)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.acrPull)
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource webAcrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(containerRegistryName)) {
  scope: containerRegistry
  name: guid(containerRegistry.id, webIdentity.id, roleIds.acrPull)
  properties: {
    roleDefinitionId: subscriptionResourceId(roleDefinitionScope, roleIds.acrPull)
    principalId: webIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('API identity resource id.')
output apiIdentityId string = apiIdentity.id

@description('API identity client id (RAG_AZURE_CLIENT_ID for the API).')
output apiIdentityClientId string = apiIdentity.properties.clientId

@description('API identity principal (object) id.')
output apiIdentityPrincipalId string = apiIdentity.properties.principalId

@description('Web identity resource id.')
output webIdentityId string = webIdentity.id

@description('Web identity client id.')
output webIdentityClientId string = webIdentity.properties.clientId

@description('Ingestion identity resource id.')
output ingestionIdentityId string = ingestionIdentity.id

@description('Ingestion identity client id (RAG_AZURE_CLIENT_ID for the Function App).')
output ingestionIdentityClientId string = ingestionIdentity.properties.clientId

@description('Ingestion identity principal (object) id.')
output ingestionIdentityPrincipalId string = ingestionIdentity.properties.principalId

@description('Qdrant identity resource id.')
output qdrantIdentityId string = qdrantIdentity.id

@description('Qdrant identity client id.')
output qdrantIdentityClientId string = qdrantIdentity.properties.clientId
