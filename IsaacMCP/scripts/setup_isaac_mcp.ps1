# IsaacMCP Setup for Windows
# Installs IsaacMCP into the main project's venv so the MCP server can run.
# Run from project root: .\IsaacMCP\scripts\setup_isaac_mcp.ps1

$ErrorActionPreference = "Stop"
$IsaacMcpDir = $PSScriptRoot | Split-Path -Parent
$ProjectRoot = $IsaacMcpDir | Split-Path -Parent
$MainVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "IsaacMCP Setup (Windows)" -ForegroundColor Cyan
Write-Host "  Project root: $ProjectRoot"
Write-Host ""

if (-not (Test-Path $MainVenvPython)) {
    Write-Host "ERROR: Main project venv not found at $MainVenvPython" -ForegroundColor Red
    Write-Host "Run .\scripts\setup_dev_env.ps1 first to create the project venv." -ForegroundColor Yellow
    exit 1
}

Write-Host "Installing IsaacMCP into main project venv..." -ForegroundColor Yellow
Push-Location $ProjectRoot
try {
    & $MainVenvPython -m pip install -e "IsaacMCP[dev]"
    Write-Host ""
    Write-Host "IsaacMCP installed. Restart Cursor to connect the MCP server." -ForegroundColor Green
} finally {
    Pop-Location
}
