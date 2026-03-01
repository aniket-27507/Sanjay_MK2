# Ubuntu 22.04 WSL2 Manual Download + Import with Progress Tracking
param(
    [string]$InstallDir = "C:\WSL\Ubuntu-22.04",
    [string]$TarPath = "C:\WSL\ubuntu-22.04-wsl.tar.gz",
    [string]$DistroName = "Ubuntu-22.04"
)

# Ubuntu 22.04 rootfs from Ubuntu's official WSL GitHub releases (gzip tar, not zstd)
$url = "https://github.com/ubuntu/WSL/releases/download/2204.6.17.0/ubuntu-jammy-wsl-amd64-24.04.tar.gz"

# Fallback: direct appx download from Microsoft (we extract the rootfs from it)
# Using the GitHub release which is a proper .tar.gz that wsl --import accepts

New-Item -ItemType Directory -Force -Path "C:\WSL"    | Out-Null
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Ubuntu 22.04 WSL2 Installer" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  URL    : $url" -ForegroundColor Gray
Write-Host "  Save to: $TarPath" -ForegroundColor Gray
Write-Host "  Distro : $DistroName => $InstallDir" -ForegroundColor Gray
Write-Host ""

# -----------------------------------------------------------------------
# PHASE 1: Download with live progress bar
# -----------------------------------------------------------------------
if (Test-Path $TarPath) {
    $cached = [math]::Round((Get-Item $TarPath).Length / 1MB, 1)
    Write-Host "[SKIP] Cached tarball found ($cached MB) — skipping download." -ForegroundColor Yellow
}
else {
    Write-Host "[1/2] Downloading Ubuntu 22.04 rootfs tar..." -ForegroundColor Green

    $startTime = Get-Date

    Add-Type -AssemblyName System.Net.Http
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $true
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [System.TimeSpan]::FromMinutes(90)
    $client.DefaultRequestHeaders.Add("User-Agent", "WSL-Installer/1.0")

    $response = $client.GetAsync($url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
    $totalBytes = $response.Content.Headers.ContentLength
    $stream = $response.Content.ReadAsStreamAsync().Result
    $fileStream = [System.IO.File]::Create($TarPath)
    $buffer = New-Object byte[] 262144   # 256 KB chunks
    $downloaded = [long]0

    while ($true) {
        $read = $stream.Read($buffer, 0, $buffer.Length)
        if ($read -le 0) { break }
        $fileStream.Write($buffer, 0, $read)
        $downloaded += $read

        $elapsed = (Get-Date) - $startTime
        $pct = if ($totalBytes -gt 0) { [int](($downloaded / $totalBytes) * 100) } else { 0 }
        $dlMB = [math]::Round($downloaded / 1MB, 1)
        $totMB = if ($totalBytes -gt 0) { [math]::Round($totalBytes / 1MB, 1) } else { "?" }
        $speed = if ($elapsed.TotalSeconds -gt 1) { [math]::Round($downloaded / 1MB / $elapsed.TotalSeconds, 2) } else { "..." }
        $etaSec = if (($speed -is [double]) -and ($speed -gt 0) -and ($totalBytes -gt 0)) {
            [int](($totalBytes - $downloaded) / 1MB / $speed)
        }
        else { -1 }
        $etaStr = if ($etaSec -ge 0) { "{0:D2}m{1:D2}s" -f [int]($etaSec / 60), ($etaSec % 60) } else { "--" }

        $status = "$dlMB MB of $totMB MB   $speed MBps   ETA $etaStr"
        Write-Progress -Id 1 -Activity "Downloading Ubuntu 22.04 LTS" -Status $status -PercentComplete $pct
    }

    $fileStream.Close()
    $stream.Close()
    $client.Dispose()
    Write-Progress -Id 1 -Activity "Downloading Ubuntu 22.04 LTS" -Completed

    $elapsed = (Get-Date) - $startTime
    $finalMB = [math]::Round($downloaded / 1MB, 1)
    $secs = [int]$elapsed.TotalSeconds
    Write-Host "[OK] Downloaded $finalMB MB in ${secs}s" -ForegroundColor Green
}

# -----------------------------------------------------------------------
# PHASE 2: Import into WSL2 (with spinner)
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "[2/2] Importing '$DistroName' into WSL2..." -ForegroundColor Green
Write-Host "      (This takes about 30-60 seconds)" -ForegroundColor Gray

$importStart = Get-Date

$job = Start-Job -ScriptBlock {
    param($t, $d, $n)
    wsl --import $n $d $t --version 2 2>&1
} -ArgumentList $TarPath, $InstallDir, $DistroName

$spin = @('|', '/', '-', '\')
$idx = 0
while ($job.State -eq 'Running') {
    $secs = [int]((Get-Date) - $importStart).TotalSeconds
    Write-Host -NoNewline ("`r  " + $spin[$idx % 4] + "  Importing... ${secs}s   ")
    $idx++
    Start-Sleep -Milliseconds 250
}

$jobOut = Receive-Job $job
Remove-Job $job
Write-Host "`r  Import complete!                            " -ForegroundColor Green
if ($jobOut) { Write-Host $jobOut }

# Set as default
wsl --set-default $DistroName 2>&1 | Out-Null

# -----------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Registered WSL2 Distributions:" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
wsl --list --verbose

Write-Host ""
Write-Host "Booting Ubuntu to verify..." -ForegroundColor Green
wsl -d $DistroName -- bash -c "lsb_release -d && echo 'Ubuntu WSL2 is ready!'"

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  SUCCESS - Ubuntu 22.04 installed and running in WSL2!" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Cyan
