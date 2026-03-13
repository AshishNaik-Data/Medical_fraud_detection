param(
    [switch]$LoadNeo4j,
    [switch]$InstallOptional,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

& (Join-Path $projectRoot "setup_local.ps1") -InstallOptional:$InstallOptional -ForceReinstall:$ForceReinstall

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found after setup."
}

$commands = @(
    @("src\data_processing.py"),
    @("src\feature_engineering.py"),
    @("src\train_model.py"),
    @("src\evaluate_model.py"),
    @("src\explainability.py"),
    @("src\neo4j_loader.py")
)

if ($LoadNeo4j) {
    $commands[-1] = @("src\neo4j_loader.py", "--load")
}

Push-Location $projectRoot
try {
    foreach ($command in $commands) {
        Write-Host ("Running " + ($command -join " "))
        & $venvPython @command
    }
}
finally {
    Pop-Location
}

Write-Host "Pipeline completed."
