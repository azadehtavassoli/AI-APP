$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pidsFile = Join-Path $root "logs\runtime\stack-pids.json"

if (-not (Test-Path $pidsFile)) {
    Write-Host "No stack pid file found at $pidsFile" -ForegroundColor Yellow
    exit 0
}

$payload = Get-Content -Path $pidsFile -Raw | ConvertFrom-Json

foreach ($proc in $payload.processes) {
    try {
        $p = Get-Process -Id $proc.pid -ErrorAction Stop
        Stop-Process -Id $p.Id -Force -ErrorAction Stop
        Write-Host "Stopped $($proc.name) (PID $($proc.pid))" -ForegroundColor Green
    }
    catch {
        Write-Host "Process $($proc.name) (PID $($proc.pid)) already stopped." -ForegroundColor Yellow
    }
}

Remove-Item -Path $pidsFile -Force
Write-Host "Stack stopped." -ForegroundColor Green
