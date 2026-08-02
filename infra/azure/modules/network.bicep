// ---------------------------------------------------------------------------------
// network.bicep — VNet, subnets and the PostgreSQL private DNS zone.
//
// Three subnets:
//   * infrastructure  — delegated to Microsoft.App/environments, hosts the Container
//                       Apps managed environment. Must be at least /23.
//   * postgres        — delegated to Microsoft.DBforPostgreSQL/flexibleServers so the
//                       Flexible Server has *private access only* (no public endpoint).
//   * privateEndpoint — spare subnet for Key Vault / Storage private endpoints.
//
// The private DNS zone is named after the server (`<server>.private.postgres.…`) which
// is the form Azure Database for PostgreSQL private access requires.
// ---------------------------------------------------------------------------------

@description('Virtual network name.')
param virtualNetworkName string

@description('Azure region for every resource in this module.')
param location string

@description('Tags applied to every resource in this module.')
param tags object = {}

@description('Address space of the virtual network.')
param vnetAddressPrefix string = '10.60.0.0/16'

@description('Subnet for the Container Apps managed environment. Must be /23 or larger.')
param containerAppsSubnetPrefix string = '10.60.0.0/23'

@description('Delegated subnet for the PostgreSQL Flexible Server (private access).')
param postgresSubnetPrefix string = '10.60.4.0/24'

@description('Subnet reserved for private endpoints (Key Vault, Storage).')
param privateEndpointSubnetPrefix string = '10.60.5.0/24'

@description('Delegated subnet for Flex Consumption Function App VNet integration.')
param functionsSubnetPrefix string = '10.60.6.0/24'

@description('PostgreSQL server name; the private DNS zone is derived from it.')
param postgresServerName string

var containerAppsSubnetName = 'snet-containerapps'
var postgresSubnetName = 'snet-postgres'
var privateEndpointSubnetName = 'snet-private-endpoints'
var functionsSubnetName = 'snet-functions'
var postgresPrivateDnsZoneName = '${postgresServerName}.private.postgres.database.azure.com'

resource containerAppsNsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: 'nsg-${containerAppsSubnetName}'
  location: location
  tags: tags
  properties: {
    securityRules: []
  }
}

resource postgresNsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: 'nsg-${postgresSubnetName}'
  location: location
  tags: tags
  properties: {
    securityRules: []
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: virtualNetworkName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: [
      {
        name: containerAppsSubnetName
        properties: {
          addressPrefix: containerAppsSubnetPrefix
          networkSecurityGroup: {
            id: containerAppsNsg.id
          }
          delegations: [
            {
              name: 'containerapps-delegation'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
          serviceEndpoints: [
            {
              service: 'Microsoft.Storage'
            }
            {
              service: 'Microsoft.KeyVault'
            }
          ]
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: postgresSubnetName
        properties: {
          addressPrefix: postgresSubnetPrefix
          networkSecurityGroup: {
            id: postgresNsg.id
          }
          delegations: [
            {
              name: 'postgres-delegation'
              properties: {
                serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
              }
            }
          ]
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: privateEndpointSubnetName
        properties: {
          addressPrefix: privateEndpointSubnetPrefix
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        // Flex Consumption VNet integration also delegates to Microsoft.App/environments,
        // and the delegated subnet cannot be shared with the Container Apps environment.
        name: functionsSubnetName
        properties: {
          addressPrefix: functionsSubnetPrefix
          delegations: [
            {
              name: 'functions-delegation'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
          serviceEndpoints: [
            {
              service: 'Microsoft.Storage'
            }
            {
              service: 'Microsoft.KeyVault'
            }
          ]
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
    ]
  }
}

resource postgresPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: postgresPrivateDnsZoneName
  location: 'global'
  tags: tags
}

resource postgresPrivateDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: postgresPrivateDnsZone
  name: 'link-${virtualNetworkName}'
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

@description('Resource id of the virtual network.')
output virtualNetworkId string = vnet.id

@description('Resource id of the Container Apps infrastructure subnet.')
output containerAppsSubnetId string = '${vnet.id}/subnets/${containerAppsSubnetName}'

@description('Resource id of the delegated PostgreSQL subnet.')
output postgresSubnetId string = '${vnet.id}/subnets/${postgresSubnetName}'

@description('Resource id of the private-endpoint subnet.')
output privateEndpointSubnetId string = '${vnet.id}/subnets/${privateEndpointSubnetName}'

@description('Resource id of the delegated Function App integration subnet.')
output functionsSubnetId string = '${vnet.id}/subnets/${functionsSubnetName}'

@description('Resource id of the PostgreSQL private DNS zone.')
output postgresPrivateDnsZoneId string = postgresPrivateDnsZone.id

@description('Name of the PostgreSQL private DNS zone.')
output postgresPrivateDnsZoneName string = postgresPrivateDnsZoneName
