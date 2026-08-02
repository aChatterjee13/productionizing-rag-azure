// ---------------------------------------------------------------------------------
// observability.bicep — Log Analytics, Application Insights and diagnostic settings.
//
// Invoked twice from main.bicep, which is how the dependency cycle is broken:
//
//   1. deployCore = true   -> creates the workspace + Application Insights. Runs first
//                             because the Container Apps environment and the Function
//                             App need them.
//   2. deployCore = false  -> creates only the diagnostic settings, once the Container
//                             Apps environment and the Function App exist.
//
// Nothing here emits a secret: the Application Insights connection string is read by
// consumers through an `existing` reference rather than passed around as an output.
// ---------------------------------------------------------------------------------

@description('Log Analytics workspace name. Created when deployCore is true, referenced otherwise.')
param logAnalyticsWorkspaceName string

@description('Application Insights component name.')
param appInsightsName string

@description('Azure region for the workspace and the Application Insights component.')
param location string

@description('Tags applied to every resource in this module.')
param tags object = {}

@description('Log retention in days for the workspace.')
@minValue(30)
@maxValue(730)
param logRetentionDays int = 30

@description('Daily ingestion cap in GB. -1 means uncapped.')
param dailyQuotaGb int = -1

@description('Create the workspace and Application Insights. False wires diagnostics only.')
param deployCore bool = true

@description('Container Apps managed environment name to attach diagnostics to. Empty skips it.')
param containerAppsEnvironmentName string = ''

@description('Function App name to attach diagnostics to. Empty skips it.')
param functionAppName string = ''

var workspaceResourceId = resourceId(
  'Microsoft.OperationalInsights/workspaces',
  logAnalyticsWorkspaceName
)

// Placeholder names keep the `existing` references compilable when the corresponding
// diagnostic setting is switched off; the settings themselves are conditional so the
// placeholder is never resolved at deploy time.
var containerAppsEnvironmentRefName = empty(containerAppsEnvironmentName)
  ? 'placeholder-container-apps-environment'
  : containerAppsEnvironmentName
var functionAppRefName = empty(functionAppName) ? 'placeholder-function-app' : functionAppName

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = if (deployCore) {
  name: logAnalyticsWorkspaceName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: logRetentionDays
    workspaceCapping: {
      dailyQuotaGb: dailyQuotaGb
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = if (deployCore) {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspaceResourceId
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    DisableLocalAuth: false
  }
  dependsOn: [
    workspace
  ]
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: containerAppsEnvironmentRefName
}

resource containerAppsDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(containerAppsEnvironmentName)) {
  name: 'diag-log-analytics'
  scope: containerAppsEnvironment
  properties: {
    workspaceId: workspaceResourceId
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
  }
}

resource functionApp 'Microsoft.Web/sites@2023-12-01' existing = {
  name: functionAppRefName
}

resource functionAppDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(functionAppName)) {
  name: 'diag-log-analytics'
  scope: functionApp
  properties: {
    workspaceId: workspaceResourceId
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

@description('Log Analytics workspace name.')
output logAnalyticsWorkspaceName string = logAnalyticsWorkspaceName

@description('Log Analytics workspace resource id.')
output logAnalyticsWorkspaceId string = workspaceResourceId

@description('Application Insights component name. Consumers read the connection string via an existing reference.')
output appInsightsName string = appInsightsName

@description('Application Insights resource id.')
output appInsightsId string = resourceId('Microsoft.Insights/components', appInsightsName)
