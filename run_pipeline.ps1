param(
    [switch]$LoadNeo4j,
    [switch]$InstallOptional,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = Join-Path $projectRoot "medical_fraud_detection"

Push-Location $appRoot
try {
    & (Join-Path $appRoot "run_pipeline.ps1") -LoadNeo4j:$LoadNeo4j -InstallOptional:$InstallOptional -ForceReinstall:$ForceReinstall
}
finally {
    Pop-Location
}
