for ($i = 1; $i -le 40; $i++) {
    $list = wsl --list 2>&1
    $time = Get-Date -Format "HH:mm:ss"
    Write-Host "$time - Attempt ${i}: $($list -join ', ')"
    if ($list -match "Ubuntu") {
        Write-Host "Ubuntu is now registered!"
        wsl --list --verbose
        exit 0
    }
    Start-Sleep -Seconds 15
}
Write-Host "Timed out waiting for Ubuntu."
