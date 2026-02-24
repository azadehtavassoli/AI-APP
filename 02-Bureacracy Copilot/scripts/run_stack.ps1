param(
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
$logsDir = Join-Path $root "logs\runtime"
$pidsFile = Join-Path $logsDir "stack-pids.json"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

if ($ForceRestart -and (Test-Path $pidsFile)) {
    & (Join-Path $PSScriptRoot "stop_stack.ps1")
}

if (Test-Path $pidsFile) {
    Write-Host "Stack seems already started (found $pidsFile). Use -ForceRestart to restart." -ForegroundColor Yellow
    exit 0
}

# Enable MCP-backed agent tools in backend process.
$env:AGENT_USE_KNOWLEDGE_MCP = "true"
$env:KNOWLEDGE_MCP_URL = "http://127.0.0.1:8765/mcp"
$env:KNOWLEDGE_MCP_TRANSPORT = "http"
$env:KNOWLEDGE_MCP_HOST = "127.0.0.1"
$env:KNOWLEDGE_MCP_PORT = "8765"

$mcpOut = Join-Path $logsDir "knowledge_mcp.out.log"
$mcpErr = Join-Path $logsDir "knowledge_mcp.err.log"
$backendOut = Join-Path $logsDir "backend.out.log"
$backendErr = Join-Path $logsDir "backend.err.log"
$frontendOut = Join-Path $logsDir "frontend.out.log"
$frontendErr = Join-Path $logsDir "frontend.err.log"

$mcp = Start-Process -FilePath $pythonExe -ArgumentList "-m knowledge_mcp.server --transport http" -WorkingDirectory $root -PassThru -WindowStyle Hidden -RedirectStandardOutput $mcpOut -RedirectStandardError $mcpErr
$backend = Start-Process -FilePath $pythonExe -ArgumentList "-m uvicorn main:app --host 127.0.0.1 --port 8000" -WorkingDirectory (Join-Path $root "backend") -PassThru -WindowStyle Hidden -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr
$frontend = Start-Process -FilePath $pythonExe -ArgumentList "-m streamlit run app.py --server.port 8501 --server.headless true" -WorkingDirectory (Join-Path $root "frontend") -PassThru -WindowStyle Hidden -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr

$payload = @{
    started_at = (Get-Date).ToString("o")
    processes = @(
        @{ name = "knowledge_mcp"; pid = $mcp.Id; stdout = $mcpOut; stderr = $mcpErr },
        @{ name = "backend"; pid = $backend.Id; stdout = $backendOut; stderr = $backendErr },
        @{ name = "frontend"; pid = $frontend.Id; stdout = $frontendOut; stderr = $frontendErr }
    )
}

$payload | ConvertTo-Json -Depth 5 | Set-Content -Path $pidsFile -Encoding UTF8

Write-Host "Started stack successfully:" -ForegroundColor Green
Write-Host "- knowledge_mcp (PID $($mcp.Id)): http://127.0.0.1:8765/mcp"
Write-Host "- backend      (PID $($backend.Id)): http://127.0.0.1:8000"
Write-Host "- frontend     (PID $($frontend.Id)): http://127.0.0.1:8501"
Write-Host "Logs: $logsDir"
