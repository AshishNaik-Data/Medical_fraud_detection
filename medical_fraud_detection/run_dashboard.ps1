param(
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

Push-Location $projectRoot
try {
    & $venvPython -m streamlit run dashboard\streamlit_app.py
}
finally {
    Pop-Location
}
