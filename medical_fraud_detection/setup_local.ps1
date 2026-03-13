param(
    [switch]$InstallOptional,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$bootstrapDir = Join-Path $projectRoot "tools\bootstrap"
$requirementsState = Join-Path $venvPath ".requirements-state.json"

function Get-ProjectPython {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        try {
            $resolved = & $pyCommand.Source -3 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return $resolved.Trim()
            }
        }
        catch {
        }
    }

    $knownLocations = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe")
    )
    foreach ($candidate in $knownLocations) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Install-PythonIfMissing {
    $pythonExe = Get-ProjectPython
    if ($pythonExe) {
        return $pythonExe
    }

    New-Item -ItemType Directory -Force -Path $bootstrapDir | Out-Null
    $installer = Join-Path $bootstrapDir "python-3.12.0-amd64.exe"
    if (-not (Test-Path $installer)) {
        Write-Host "Downloading Python 3.12 installer..."
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe" -OutFile $installer
    }

    Write-Host "Installing Python 3.12 for the current user..."
    Start-Process -FilePath $installer -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_launcher=1","Include_pip=1","Shortcuts=0" -Wait
    Start-Sleep -Seconds 5

    $pythonExe = Get-ProjectPython
    if (-not $pythonExe) {
        throw "Python installation finished, but python.exe was not found. Reopen PowerShell and rerun .\setup_local.ps1"
    }
    return $pythonExe
}

function New-ProjectVenv {
    param(
        [string]$PythonExe,
        [string]$TargetPath
    )

    if (Test-Path $TargetPath) {
        Remove-Item $TargetPath -Recurse -Force
    }

    Write-Host "Creating virtual environment in $TargetPath"
    & $PythonExe -m venv $TargetPath
    if (-not (Test-Path (Join-Path $TargetPath "Scripts\python.exe"))) {
        throw "Virtual environment creation failed."
    }
}

function Get-FileHashString {
    param(
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return ""
    }
    return (Get-FileHash $Path -Algorithm SHA256).Hash
}

function Test-RequirementsState {
    param(
        [string]$StatePath,
        [string]$RequirementsHash,
        [string]$OptionalHash,
        [bool]$OptionalInstalled
    )

    if (-not (Test-Path $StatePath)) {
        return $false
    }

    try {
        $state = Get-Content $StatePath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }

    if ($state.requirements_hash -ne $RequirementsHash) {
        return $false
    }
    if ($OptionalInstalled -and $state.optional_hash -ne $OptionalHash) {
        return $false
    }
    if (-not $OptionalInstalled -and $state.optional_installed) {
        return $true
    }
    return $true
}

function Test-CoreImports {
    param(
        [string]$PythonExe
    )

    & $PythonExe -c "import pandas,numpy,sklearn,streamlit,yaml,joblib,networkx,matplotlib; print('ok')" *> $null
    return $LASTEXITCODE -eq 0
}

function Test-OptionalImports {
    param(
        [string]$PythonExe
    )

    & $PythonExe -c "import lightgbm,shap,neo4j,py2neo; print('ok')" *> $null
    return $LASTEXITCODE -eq 0
}

function Sync-Requirements {
    param(
        [string]$PythonExe,
        [bool]$OptionalInstalled,
        [bool]$ForceInstall
    )

    $requirementsPath = Join-Path $projectRoot "requirements.txt"
    $optionalRequirementsPath = Join-Path $projectRoot "requirements-optional.txt"
    $requirementsHash = Get-FileHashString -Path $requirementsPath
    $optionalHash = Get-FileHashString -Path $optionalRequirementsPath

    $needsInstall = $ForceInstall -or -not (Test-RequirementsState -StatePath $requirementsState -RequirementsHash $requirementsHash -OptionalHash $optionalHash -OptionalInstalled $OptionalInstalled)

    if (-not (Test-CoreImports -PythonExe $PythonExe)) {
        $needsInstall = $true
    }
    if ($OptionalInstalled -and -not (Test-OptionalImports -PythonExe $PythonExe)) {
        $needsInstall = $true
    }

    if (-not $needsInstall) {
        Write-Host "Dependencies already satisfied."
        return
    }

    Write-Host "Installing project dependencies..."
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r $requirementsPath
    if ($OptionalInstalled) {
        & $PythonExe -m pip install -r $optionalRequirementsPath
    }

    $state = @{
        requirements_hash = $requirementsHash
        optional_hash = $optionalHash
        optional_installed = $OptionalInstalled
    } | ConvertTo-Json
    Set-Content -Path $requirementsState -Value $state -Encoding UTF8
}

$pythonExe = Install-PythonIfMissing

if ($ForceReinstall -or -not (Test-Path $venvPython)) {
    New-ProjectVenv -PythonExe $pythonExe -TargetPath $venvPath
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment is missing after setup."
}

Push-Location $projectRoot
try {
    Sync-Requirements -PythonExe $venvPython -OptionalInstalled $InstallOptional.IsPresent -ForceInstall $ForceReinstall.IsPresent
}
finally {
    Pop-Location
}

Write-Host "Environment ready."
Write-Host "Run .\start_project.ps1 to open the dashboard on this machine."
