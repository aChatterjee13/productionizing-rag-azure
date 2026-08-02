// ---------------------------------------------------------------------------------
// keyvault.bicep — RBAC-mode vault holding every secret the platform needs.
//
// RBAC mode only: no access policies. Role assignments live in identity.bicep
// (Key Vault Secrets User for each workload identity, Secrets Officer for the
// deploying principal so `deploy.sh` can rotate values).
//
// A secret resource is created only when its (@secure) parameter is non-empty, so a
// re-deploy that omits an optional secret leaves the stored value untouched instead of
// overwriting it with a placeholder. Outputs are versionless secret *URIs* — never
// values — which is exactly what Container Apps `secrets[].keyVaultUrl` and Function
// App `@Microsoft.KeyVault(SecretUri=…)` references want.
// ---------------------------------------------------------------------------------

@description('Key Vault name. 3-24 alphanumerics and dashes, must start with a letter.')
@minLength(3)
@maxLength(24)
param keyVaultName string

@description('Azure region.')
param location string

@description('Tags applied to the vault.')
param tags object = {}

@description('Vault SKU.')
@allowed([
  'standard'
  'premium'
])
param skuName string = 'standard'

@description('Days a soft-deleted secret is recoverable.')
@minValue(7)
@maxValue(90)
param softDeleteRetentionInDays int = 90

@description('Enable purge protection. Irreversible once enabled — on for production.')
param enablePurgeProtection bool = false

@description('Allow public network access to the data plane.')
@allowed([
  'Enabled'
  'Disabled'
])
param publicNetworkAccess string = 'Enabled'

@description('Anthropic API key. Empty leaves any existing secret untouched.')
@secure()
param anthropicApiKey string = ''

@description('Qdrant API key. Empty leaves any existing secret untouched.')
@secure()
param qdrantApiKey string = ''

@description('PostgreSQL administrator password. Empty leaves any existing secret untouched.')
@secure()
param postgresAdminPassword string = ''

@description('Langfuse public key. Empty leaves any existing secret untouched.')
@secure()
param langfusePublicKey string = ''

@description('Langfuse secret key. Empty leaves any existing secret untouched.')
@secure()
param langfuseSecretKey string = ''

@description('HMAC key for pii_redaction_mode="hash". Empty leaves any existing secret untouched.')
@secure()
param piiHashSecret string = ''

var secretNames = {
  anthropicApiKey: 'anthropic-api-key'
  qdrantApiKey: 'qdrant-api-key'
  postgresAdminPassword: 'postgres-admin-password'
  langfusePublicKey: 'langfuse-public-key'
  langfuseSecretKey: 'langfuse-secret-key'
  piiHashSecret: 'pii-hash-secret'
}

var secretDefinitions = [
  {
    name: secretNames.anthropicApiKey
    value: anthropicApiKey
    contentType: 'Anthropic API key (RAG_ANTHROPIC_API_KEY)'
  }
  {
    name: secretNames.qdrantApiKey
    value: qdrantApiKey
    contentType: 'Qdrant API key (RAG_QDRANT_API_KEY / QDRANT__SERVICE__API_KEY)'
  }
  {
    name: secretNames.postgresAdminPassword
    value: postgresAdminPassword
    contentType: 'PostgreSQL administrator password (RAG_POSTGRES_PASSWORD)'
  }
  {
    name: secretNames.langfusePublicKey
    value: langfusePublicKey
    contentType: 'Langfuse public key (RAG_LANGFUSE_PUBLIC_KEY)'
  }
  {
    name: secretNames.langfuseSecretKey
    value: langfuseSecretKey
    contentType: 'Langfuse secret key (RAG_LANGFUSE_SECRET_KEY)'
  }
  {
    name: secretNames.piiHashSecret
    value: piiHashSecret
    contentType: 'HMAC key for hashed PII redaction (RAG_PII_HASH_SECRET)'
  }
]

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: skuName
    }
    // RBAC only. Nothing in this repository writes access policies.
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: softDeleteRetentionInDays
    // Must be null rather than false: Azure rejects an explicit false once enabled.
    enablePurgeProtection: enablePurgeProtection ? true : null
    publicNetworkAccess: publicNetworkAccess
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

resource secrets 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = [
  for definition in secretDefinitions: if (!empty(definition.value)) {
    parent: keyVault
    name: definition.name
    properties: {
      value: definition.value
      contentType: definition.contentType
      attributes: {
        enabled: true
      }
    }
  }
]

@description('Key Vault name.')
output keyVaultName string = keyVault.name

@description('Key Vault resource id.')
output keyVaultId string = keyVault.id

@description('Key Vault data-plane URI, with trailing slash.')
output keyVaultUri string = keyVault.properties.vaultUri

@description('Secret names, so consumers never hard-code them.')
output secretNames object = secretNames

@description('Versionless secret URIs for Container Apps and Function App references.')
output secretUris object = {
  anthropicApiKey: '${keyVault.properties.vaultUri}secrets/${secretNames.anthropicApiKey}'
  qdrantApiKey: '${keyVault.properties.vaultUri}secrets/${secretNames.qdrantApiKey}'
  postgresAdminPassword: '${keyVault.properties.vaultUri}secrets/${secretNames.postgresAdminPassword}'
  langfusePublicKey: '${keyVault.properties.vaultUri}secrets/${secretNames.langfusePublicKey}'
  langfuseSecretKey: '${keyVault.properties.vaultUri}secrets/${secretNames.langfuseSecretKey}'
  piiHashSecret: '${keyVault.properties.vaultUri}secrets/${secretNames.piiHashSecret}'
}
