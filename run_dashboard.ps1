param(
    [switch]$InstallOptional,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = Join-Path $projectRoot "medical_fraud_detection"

Push-Location $appRoot
try {
    & (Join-Path $appRoot "run_dashboard.ps1") -InstallOptional:$InstallOptional -ForceReinstall:$ForceReinstall
}
finally {
    Pop-Location
}
