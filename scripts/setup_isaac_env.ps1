# =============================================================================
# Project Sanjay Mk2 - Windows Isaac Sim Environment Setup
# =============================================================================
# Run this in PowerShell BEFORE launching Isaac Sim.
#
# Usage:
#   .\scripts\setup_isaac_env.ps1
#   .\scripts\setup_isaac_env.ps1 -DomainId 10
#
# This sets session-only environment variables. For permanent setup,
# use System Properties > Environment Variables.
# =============================================================================

param(
    [string]$FastDDSProfilePath = "$PSScriptRoot\..\network\fastdds_profiles.xml",
    [int]$DomainId = 10
)

# Resolve the profile path
$profilePath = Resolve-Path $FastDDSProfilePath -ErrorAction SilentlyContinue
if (-not $profilePath) {
    $profilePath = "$env:USERPROFILE\fastdds_profiles.xml"
    Write-Host "Profile not found at original path, using: $profilePath"
}

# Set session environment variables
$env:ROS_DOMAIN_ID = $DomainId.ToString()
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
$env:FASTRTPS_DEFAULT_PROFILES_FILE = $profilePath
$env:ROS_LOCALHOST_ONLY = "0"

Write-Host "Environment variables set for current session:" -ForegroundColor Green
Write-Host "  ROS_DOMAIN_ID = $($env:ROS_DOMAIN_ID)"
Write-Host "  RMW_IMPLEMENTATION = $($env:RMW_IMPLEMENTATION)"
Write-Host "  FASTRTPS_DEFAULT_PROFILES_FILE = $($env:FASTRTPS_DEFAULT_PROFILES_FILE)"
Write-Host "  ROS_LOCALHOST_ONLY = $($env:ROS_LOCALHOST_ONLY)"
Write-Host ""
Write-Host "Restart Isaac Sim for changes to take effect." -ForegroundColor Yellow
