# =============================================================================
# Project Sanjay Mk2 - Development Environment Setup (Windows)
# =============================================================================
#
# Creates a Python 3.11 virtual environment with all pinned dependencies.
# Run this once after cloning the repo.
#
# Usage:
#   .\scripts\setup_dev_env.ps1
#
# Prerequisites:
#   - Python 3.11 installed (via python.org or py launcher)
#   - If not installed, this script will attempt to download and install it
#
# =============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPath = Join-Path $ProjectRoot ".venv"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$PythonVersion = "3.11"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Project Sanjay Mk2 - Dev Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Find or install Python 3.11 ──────────────────────────
Write-Host "[1/5] Checking for Python $PythonVersion..." -ForegroundColor Yellow

$py311 = $null

# Try py launcher first
try {
    $pyList = py --list 2>&1
    if ($pyList -match "3\.11") {
        $py311 = "py -$PythonVersion"
        $version = & py -3.11 --version 2>&1
        Write-Host "  Found via py launcher: $version" -ForegroundColor Green
    }
} catch {}

# Try direct python3.11 command
if (-not $py311) {
    try {
        $version = & python3.11 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $py311 = "python3.11"
            Write-Host "  Found: $version" -ForegroundColor Green
        }
    } catch {}
}

# Try common install locations
if (-not $py311) {
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Python311\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($p in $commonPaths) {
        if (Test-Path $p) {
            $py311 = $p
            $version = & $p --version 2>&1
            Write-Host "  Found at: $p ($version)" -ForegroundColor Green
            break
        }
    }
}

# Not found — offer to install
if (-not $py311) {
    Write-Host "  Python $PythonVersion not found." -ForegroundColor Red
    Write-Host ""
    $answer = Read-Host "  Download and install Python 3.11.7? (Y/n)"
    if ($answer -eq "" -or $answer -eq "Y" -or $answer -eq "y") {
        Write-Host "  Downloading Python 3.11.7..." -ForegroundColor Yellow
        $installerUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
        $installerPath = Join-Path $env:TEMP "python-3.11.7-amd64.exe"
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing

        Write-Host "  Installing Python 3.11.7..." -ForegroundColor Yellow
        $proc = Start-Process -FilePath $installerPath -ArgumentList `
            "/quiet", "InstallAllUsers=0", "PrependPath=0", `
            "Include_launcher=1", "Include_test=0" `
            -Wait -PassThru

        if ($proc.ExitCode -ne 0) {
            Write-Host "  Installation failed (exit code $($proc.ExitCode))" -ForegroundColor Red
            exit 1
        }

        $py311 = "py -$PythonVersion"
        Write-Host "  Installed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "  Please install Python 3.11 from https://python.org/downloads/" -ForegroundColor Red
        exit 1
    }
}

# ── Step 2: Create virtual environment ──────────────────────────
Write-Host ""
Write-Host "[2/5] Creating virtual environment at .venv/..." -ForegroundColor Yellow

if (Test-Path $VenvPath) {
    # Check if existing venv has correct Python version
    $existingPy = & "$VenvPath\Scripts\python.exe" --version 2>&1
    if ($existingPy -match "3\.11") {
        Write-Host "  Existing venv with Python 3.11 found, reusing." -ForegroundColor Green
    } else {
        Write-Host "  Existing venv has wrong Python ($existingPy), recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvPath
        if ($py311 -match "^py ") {
            & py -3.11 -m venv $VenvPath
        } else {
            & $py311 -m venv $VenvPath
        }
    }
} else {
    if ($py311 -match "^py ") {
        & py -3.11 -m venv $VenvPath
    } else {
        & $py311 -m venv $VenvPath
    }
}
Write-Host "  Done." -ForegroundColor Green

# ── Step 3: Upgrade pip ──────────────────────────────────────────
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip --quiet
Write-Host "  Done." -ForegroundColor Green

# ── Step 4: Install dependencies ─────────────────────────────────
Write-Host ""
Write-Host "[4/5] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& "$VenvPath\Scripts\pip.exe" install -r $RequirementsPath --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Some packages failed to install. Check output above." -ForegroundColor Red
    exit 1
}
Write-Host "  Done." -ForegroundColor Green

# ── Step 5: Verify ────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow

$pyVersion = & "$VenvPath\Scripts\python.exe" --version 2>&1
Write-Host "  Python:     $pyVersion" -ForegroundColor White

$numpyVer = & "$VenvPath\Scripts\python.exe" -c "import numpy; print(numpy.__version__)" 2>&1
Write-Host "  numpy:      $numpyVer" -ForegroundColor White

$torchVer = & "$VenvPath\Scripts\python.exe" -c "import torch; print(torch.__version__)" 2>&1
Write-Host "  torch:      $torchVer" -ForegroundColor White

$mujocoVer = & "$VenvPath\Scripts\python.exe" -c "import mujoco; print(mujoco.__version__)" 2>&1
Write-Host "  mujoco:     $mujocoVer" -ForegroundColor White

$mavsdkVer = & "$VenvPath\Scripts\python.exe" -c "import mavsdk; print('OK')" 2>&1
Write-Host "  mavsdk:     $mavsdkVer" -ForegroundColor White

$pkgCount = & "$VenvPath\Scripts\pip.exe" list --format=columns 2>&1 | Measure-Object -Line
Write-Host "  Total pkgs: $($pkgCount.Lines - 2)" -ForegroundColor White

# ── Done ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate the environment:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Run tests:" -ForegroundColor White
Write-Host "    python -m pytest tests/ -v" -ForegroundColor Cyan
Write-Host ""
