// ---------------------------------------------------------------------------------
// postgres.bicep — Azure Database for PostgreSQL Flexible Server, private access only.
//
// * VNet-integrated (delegated subnet + private DNS zone). No public endpoint, no
//   firewall rules: reaching it requires being inside the VNet, which the Container
//   Apps environment and the Function App are.
// * Entra ID administrator alongside the password admin, so operators authenticate
//   with their directory identity. Password auth stays on because the API and the
//   ingestion Function App read RAG_POSTGRES_PASSWORD from Key Vault; switching them
//   to token auth is a code change in ragcore.db, not an infrastructure change.
// * Backup retention and geo-redundancy are parameters — dev keeps 7 days, prod 35.
// ---------------------------------------------------------------------------------

@description('PostgreSQL Flexible Server name.')
param serverName string

@description('Azure region.')
param location string

@description('Tags applied to the server.')
param tags object = {}

@description('Compute SKU, e.g. Standard_B2s (burstable) or Standard_D2ds_v5.')
param skuName string = 'Standard_B2s'

@description('SKU tier matching skuName.')
@allowed([
  'Burstable'
  'GeneralPurpose'
  'MemoryOptimized'
])
param skuTier string = 'Burstable'

@description('PostgreSQL major version.')
@allowed([
  '15'
  '16'
  '17'
])
param postgresVersion string = '16'

@description('Data disk size in GiB.')
@allowed([
  32
  64
  128
  256
  512
])
param storageSizeGB int = 32

@description('Administrator login name (password auth).')
param administratorLogin string = 'ragadmin'

@description('Administrator password. Sourced from Key Vault by deploy.sh.')
@secure()
param administratorPassword string

@description('Application database name; must match RAG_POSTGRES_DB.')
param databaseName string = 'rag'

@description('Delegated subnet for private access.')
param delegatedSubnetId string

@description('Private DNS zone resource id for the server FQDN.')
param privateDnsZoneId string

@description('Point-in-time backup retention in days.')
@minValue(7)
@maxValue(35)
param backupRetentionDays int = 7

@description('Store backups in the paired region as well.')
param geoRedundantBackup bool = false

@description('Zone-redundant high availability. Requires a GeneralPurpose or higher SKU.')
param zoneRedundantHighAvailability bool = false

@description('Entra ID object id of the directory administrator. Empty skips the Entra admin.')
param entraAdminObjectId string = ''

@description('Display name / UPN of the Entra administrator.')
param entraAdminPrincipalName string = ''

@description('Principal type of the Entra administrator.')
@allowed([
  'User'
  'Group'
  'ServicePrincipal'
])
param entraAdminPrincipalType string = 'User'

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: serverName
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: skuTier
  }
  properties: {
    version: postgresVersion
    createMode: 'Default'
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorPassword
    authConfig: {
      activeDirectoryAuth: empty(entraAdminObjectId) ? 'Disabled' : 'Enabled'
      passwordAuth: 'Enabled'
      tenantId: subscription().tenantId
    }
    storage: {
      storageSizeGB: storageSizeGB
      autoGrow: 'Enabled'
    }
    backup: {
      backupRetentionDays: backupRetentionDays
      geoRedundantBackup: geoRedundantBackup ? 'Enabled' : 'Disabled'
    }
    highAvailability: {
      mode: zoneRedundantHighAvailability ? 'ZoneRedundant' : 'Disabled'
    }
    // Private access: no public endpoint and no firewall rules are possible.
    network: {
      delegatedSubnetResourceId: delegatedSubnetId
      privateDnsZoneArmResourceId: privateDnsZoneId
    }
    maintenanceWindow: {
      customWindow: 'Enabled'
      dayOfWeek: 0
      startHour: 3
      startMinute: 0
    }
  }
}

resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgres
  name: databaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

resource entraAdministrator 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = if (!empty(entraAdminObjectId)) {
  parent: postgres
  name: entraAdminObjectId
  properties: {
    principalType: entraAdminPrincipalType
    principalName: empty(entraAdminPrincipalName) ? entraAdminObjectId : entraAdminPrincipalName
    tenantId: subscription().tenantId
  }
  dependsOn: [
    database
  ]
}

@description('Server name.')
output serverName string = postgres.name

@description('Server resource id.')
output serverId string = postgres.id

@description('Fully qualified domain name; the value RAG_POSTGRES_HOST must carry.')
output fullyQualifiedDomainName string = postgres.properties.fullyQualifiedDomainName

@description('Application database name.')
output databaseName string = database.name

@description('Administrator login for password auth.')
output administratorLogin string = administratorLogin
