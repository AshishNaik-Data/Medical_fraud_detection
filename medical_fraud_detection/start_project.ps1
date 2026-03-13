param(
    [switch]$InstallOptional,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Preparing the project environment..."
& (Join-Path $projectRoot "run_dashboard.ps1") -InstallOptional:$InstallOptional -ForceReinstall:$ForceReinstall
