# =============================================================================
# Project Sanjay Mk2 - WSL2 Bootstrap (Windows)
# =============================================================================
#
# One-shot setup for WSL2 + Docker Desktop + Ubuntu provisioning.
#
# Usage:
#   .\scripts\bootstrap_wsl2.ps1
#
# =============================================================================

[CmdletBinding()]
param(
    [string]$DistroName = "Ubuntu-22.04"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param(
        [string]$Label,
        [string]$Message
    )
    Write-Host "[$Label] $Message" -ForegroundColor Yellow
}

function Convert-WindowsPathToWslPath {
    param([string]$WindowsPath)

    $full = (Resolve-Path $WindowsPath).Path
    $drive = $full.Substring(0, 1).ToLower()
    $tail = $full.Substring(2).Replace("\", "/")
    return "/mnt/$drive$tail"
}

function Test-WindowsFeatureEnabled {
    param([string]$FeatureName)
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $FeatureName
    return $feature.State -eq "Enabled"
}

function Enable-WindowsFeatureIfNeeded {
    param([string]$FeatureName)
    if (-not (Test-WindowsFeatureEnabled -FeatureName $FeatureName)) {
        Write-Host "  Enabling Windows feature: $FeatureName" -ForegroundColor Cyan
        Enable-WindowsOptionalFeature -Online -FeatureName $FeatureName -NoRestart | Out-Null
        return $true
    }
    Write-Host "  Feature already enabled: $FeatureName" -ForegroundColor Green
    return $false
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$WslConfigTemplate = Join-Path $ProjectRoot "config\wsl\.wslconfig"
$WslConfTemplate = Join-Path $ProjectRoot "config\wsl\wsl.conf"
$InstallUbuntuScript = Join-Path $ProjectRoot "scripts\install_ubuntu_wsl2.ps1"
$WaitUbuntuScript = Join-Path $ProjectRoot "scripts\wait_for_ubuntu.ps1"
$WindowsWslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"
$WslProjectPath = Convert-WindowsPathToWslPath -WindowsPath $ProjectRoot

if (-not (Test-Path $WslConfigTemplate)) { throw "Missing template: $WslConfigTemplate" }
if (-not (Test-Path $WslConfTemplate)) { throw "Missing template: $WslConfTemplate" }
if (-not (Test-Path $InstallUbuntuScript)) { throw "Missing script: $InstallUbuntuScript" }
if (-not (Test-Path $WaitUbuntuScript)) { throw "Missing script: $WaitUbuntuScript" }

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Project Sanjay Mk2 - WSL2 Bootstrap" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

$requiresReboot = $false

Write-Step -Label "1/8" -Message "Checking required Windows features..."
$requiresReboot = (Enable-WindowsFeatureIfNeeded -FeatureName "Microsoft-Windows-Subsystem-Linux") -or $requiresReboot
$requiresReboot = (Enable-WindowsFeatureIfNeeded -FeatureName "VirtualMachinePlatform") -or $requiresReboot

Write-Step -Label "2/8" -Message "Writing $WindowsWslConfigPath..."
Copy-Item -Path $WslConfigTemplate -Destination $WindowsWslConfigPath -Force
Write-Host "  Updated .wslconfig from template." -ForegroundColor Green

Write-Step -Label "3/8" -Message "Checking Ubuntu distro registration..."
$distroList = wsl --list --quiet 2>$null
if (-not ($distroList -match [regex]::Escape($DistroName))) {
    Write-Host "  Distro '$DistroName' not found. Installing..." -ForegroundColor Yellow
    & $InstallUbuntuScript -DistroName $DistroName
    & $WaitUbuntuScript
} else {
    Write-Host "  Distro '$DistroName' is already registered." -ForegroundColor Green
}

Write-Step -Label "4/8" -Message "Pushing /etc/wsl.conf into $DistroName..."
$wslConfContent = Get-Content -Raw -Path $WslConfTemplate
$wslConfBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($wslConfContent))
wsl -d $DistroName -u root -- bash -lc "echo '$wslConfBase64' | base64 -d > /etc/wsl.conf"
Write-Host "  /etc/wsl.conf written." -ForegroundColor Green

Write-Step -Label "5/8" -Message "Restarting WSL to apply .wslconfig and wsl.conf..."
wsl --shutdown
Start-Sleep -Seconds 2
Write-Host "  WSL shutdown complete." -ForegroundColor Green

Write-Step -Label "6/8" -Message "Running Ubuntu provisioning scripts..."
wsl -d $DistroName -u root -- bash -lc "cd '$WslProjectPath' && chmod +x scripts/setup_wsl2_env.sh scripts/validate_setup.sh && bash scripts/setup_wsl2_env.sh --as-root"

$WslUser = (wsl -d $DistroName -- bash -lc "getent passwd 1000 | cut -d: -f1").Trim()
if ([string]::IsNullOrWhiteSpace($WslUser)) {
    $WslUser = (wsl -d $DistroName -- bash -lc "id -un").Trim()
}
Write-Host "  Using WSL user: $WslUser" -ForegroundColor Gray
wsl -d $DistroName -u $WslUser -- bash -lc "cd '$WslProjectPath' && bash scripts/setup_wsl2_env.sh --as-user"

Write-Step -Label "7/8" -Message "Checking Docker Desktop WSL integration settings..."
$dockerSettingsCandidates = @(
    (Join-Path $env:APPDATA "Docker\settings-store.json"),
    (Join-Path $env:APPDATA "Docker\settings.json")
)
$dockerSettingsFound = $false
$dockerIntegrationLooksEnabled = $false

foreach ($settingsPath in $dockerSettingsCandidates) {
    if (Test-Path $settingsPath) {
        $dockerSettingsFound = $true
        $settingsRaw = Get-Content -Raw -Path $settingsPath
        $wslEngineEnabled = $settingsRaw -match '"wslEngineEnabled"\s*:\s*true'
        $distroIntegrated = $settingsRaw -match [regex]::Escape($DistroName)
        if ($wslEngineEnabled -and $distroIntegrated) {
            $dockerIntegrationLooksEnabled = $true
            Write-Host "  Docker WSL integration looks enabled in $settingsPath" -ForegroundColor Green
            break
        }
    }
}

if (-not $dockerSettingsFound) {
    Write-Host "  Docker settings file not found. Open Docker Desktop and confirm WSL integration manually." -ForegroundColor Yellow
} elseif (-not $dockerIntegrationLooksEnabled) {
    Write-Host "  Could not confirm distro integration for '$DistroName' from Docker settings." -ForegroundColor Yellow
    Write-Host "  Verify in Docker Desktop -> Settings -> Resources -> WSL Integration." -ForegroundColor Yellow
}

Write-Step -Label "8/8" -Message "Running validation script..."
wsl -d $DistroName -- bash -lc "cd '$WslProjectPath' && bash scripts/validate_setup.sh"

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  WSL2 bootstrap completed" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green

if ($requiresReboot) {
    Write-Host ""
    Write-Host "NOTE: Windows features were enabled during setup." -ForegroundColor Yellow
    Write-Host "A system reboot is recommended before running full simulations." -ForegroundColor Yellow
}
