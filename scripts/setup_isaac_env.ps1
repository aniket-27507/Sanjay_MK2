# =============================================================================
# Project Sanjay Mk2 - Isaac Sim ROS 2 Environment Setup (Windows)
# =============================================================================
# Run this BEFORE launching Isaac Sim:
#
#   .\scripts\setup_isaac_env.ps1
#
# Then launch Isaac Sim:
#   isaacsim --enable omni.isaac.ros2_bridge
#
# This script sets ALL variables Isaac Sim's bridge needs to initialise.
# =============================================================================

param(
    [int]$DomainId = 10
)

# ── Paths ─────────────────────────────────────────────────────────────────────
$VenvRoot = "D:\Sanjay_MK2\.venv"
$HumbleRoot = "$VenvRoot\Lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\humble"
$HumbleLib = "$HumbleRoot\lib"
$HumbleRcl = "$HumbleRoot\rclpy"
$FastDDS = "D:\Sanjay_MK2\network\fastdds_profiles.xml"

# ── Validate ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Isaac Sim ROS 2 Environment Setup ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $HumbleLib)) {
    Write-Host "  ERROR: Bundled ROS 2 libs not found at:" -ForegroundColor Red
    Write-Host "  $HumbleLib" -ForegroundColor Red
    Write-Host "  Run: pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com" -ForegroundColor Yellow
    exit 1
}

# ── 1. ROS 2 Core Identity ────────────────────────────────────────────────────
$env:ROS_DISTRO = "humble"
$env:ROS_VERSION = "2"
$env:ROS_PYTHON_VERSION = "3"

# ── 2. DDS / RMW Transport ───────────────────────────────────────────────────
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
$env:FASTRTPS_DEFAULT_PROFILES_FILE = $FastDDS
$env:ROS_DOMAIN_ID = $DomainId.ToString()

# ── 3. Network ────────────────────────────────────────────────────────────────
$env:ROS_LOCALHOST_ONLY = "0"

# ── 4. PATH — bundled ROS 2 DLLs must be on PATH on Windows ──────────────────
if ($env:PATH -notlike "*$HumbleLib*") {
    $env:PATH = "$HumbleLib;" + $env:PATH
}

# ── 5. PYTHONPATH — expose bundled rclpy ─────────────────────────────────────
if ($env:PYTHONPATH -notlike "*$HumbleRcl*") {
    $env:PYTHONPATH = "$HumbleRcl;" + $env:PYTHONPATH
}

# ── 6. AMENT_PREFIX_PATH — lets message packages be discovered ───────────────
$env:AMENT_PREFIX_PATH = $HumbleRoot

# ── Print summary ─────────────────────────────────────────────────────────────
Write-Host "  ROS_DISTRO                     = $($env:ROS_DISTRO)"
Write-Host "  RMW_IMPLEMENTATION             = $($env:RMW_IMPLEMENTATION)"
Write-Host "  ROS_DOMAIN_ID                  = $($env:ROS_DOMAIN_ID)"
Write-Host "  ROS_LOCALHOST_ONLY             = $($env:ROS_LOCALHOST_ONLY)"
Write-Host "  FASTRTPS_DEFAULT_PROFILES_FILE = $($env:FASTRTPS_DEFAULT_PROFILES_FILE)"
Write-Host "  AMENT_PREFIX_PATH              = $($env:AMENT_PREFIX_PATH)"
Write-Host "  Humble DLLs on PATH:           $HumbleLib"
Write-Host "  rclpy on PYTHONPATH:           $HumbleRcl"
Write-Host ""
Write-Host "Now launch Isaac Sim:" -ForegroundColor Green
Write-Host "  isaacsim --enable omni.isaac.ros2_bridge" -ForegroundColor Cyan
Write-Host ""
