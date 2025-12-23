param(
  [int]$Port = 8000,
  [string]$HostAddr = "127.0.0.1",
  [switch]$Reload,
  [string]$ArcticUri = "",
  [switch]$KillPort
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $repo
Set-Location $repo

if ($ArcticUri -ne "") {
  $env:QUANTDSL_ARCTIC_URI = $ArcticUri
}

if ($KillPort) {
  try {
    $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn -and $conn.OwningProcess) {
      Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
      Start-Sleep -Milliseconds 300
    }
  } catch {
    # ignore
  }
}

$reloadFlag = $Reload.IsPresent ? "--reload" : ""

Write-Host "Starting Platform UI on http://$HostAddr`:$Port/" -ForegroundColor Cyan
Write-Host "OpenAPI: http://$HostAddr`:$Port/docs" -ForegroundColor Cyan

# Use uv (repo standard)
if ($reloadFlag -ne "") {
  uv run python -m uvicorn quantdsl_backtest.platform_api.main:app --host $HostAddr --port $Port --log-level info $reloadFlag
} else {
  uv run python -m uvicorn quantdsl_backtest.platform_api.main:app --host $HostAddr --port $Port --log-level info
}

