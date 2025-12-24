<#
Starts the QuantDSL Backtest Platform UI server.

This is a convenience wrapper around:
  uv run python scripts\run_platform_ui.py

It writes:
- logs to:     .platform_ui/server.log
- PID to:      .platform_ui/server.pid

Usage:
  .\scripts\run_platform_ui.ps1

Notes:
- If a previous PID file exists and the process is still running, this script exits.
- To stop the server, run: .\scripts\stop_platform_ui.ps1
#>

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root '.platform_ui'
$pidFile = Join-Path $stateDir 'server.pid'
$logFile = Join-Path $stateDir 'server.log'
$portFile = Join-Path $stateDir 'server.port'

param(
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 8000
)

if (!(Test-Path $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir | Out-Null
}

# Ensure log file exists so Start-Process redirection always has a target.
if (!(Test-Path $logFile)) {
    New-Item -ItemType File -Path $logFile -Force | Out-Null
}

if (Test-Path $pidFile) {
    $oldPidText = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($oldPidText) {
        $oldPid = 0
        [void][int]::TryParse($oldPidText.Trim(), [ref]$oldPid)
        if ($oldPid -gt 0) {
            $oldProc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($null -ne $oldProc) {
                Write-Output ("Platform UI already running (PID {0}). Log: {1}" -f $oldPid, $logFile)
                exit 0
            }
        }
    }

    # Stale PID file
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host ("Starting Platform UI... host={0} port={1} log={2}" -f $HostAddress, $Port, $logFile)

# Fail fast if port is already in use.
$ownerPid = $null
try {
    # netstat output includes PID in the last column.
    $line = (netstat -aon | Select-String -Pattern (':'+$Port+'\s') | Select-Object -First 1)
    if ($null -ne $line) {
        $parts = ($line.Line -split '\s+') | Where-Object { $_ -ne '' }
        if ($parts.Count -ge 5) {
            $ownerPid = $parts[-1]
        }
    }
} catch {
    $ownerPid = $null
}

if ($null -ne $ownerPid -and $ownerPid -ne '') {
    $procName = $null
    try {
        $p = Get-Process -Id ([int]$ownerPid) -ErrorAction Stop
        $procName = $p.ProcessName
    } catch {
        $procName = $null
    }

    if ([string]::IsNullOrWhiteSpace($procName)) {
        Write-Host ("Port {0} is already in use by PID {1}." -f $Port, $ownerPid)
    } else {
        Write-Host ("Port {0} is already in use by PID {1} ({2})." -f $Port, $ownerPid, $procName)
    }

    Write-Host ("If this is a stale Platform UI server, run .\\scripts\\stop_platform_ui.ps1. Otherwise start on another port, e.g. .\\scripts\\run_platform_ui.ps1 -Port 8001" )
    throw ("Port {0} in use" -f $Port)
}

# Start in repo root so relative paths behave.
$workDir = $root

# Resolve how to invoke uv.
$uvCmd = $null
try {
    $uvCmd = (Get-Command uv -ErrorAction Stop).Source
} catch {
    $uvCmd = $null
}

if ([string]::IsNullOrWhiteSpace($uvCmd)) {
    throw "Couldn't find 'uv' on PATH. Ensure uv is installed/available, or start the server with: uv run python scripts\\run_platform_ui.py"
}

$uvArgs = @('run','python','scripts\\run_platform_ui.py','--host',$HostAddress,'--port',[string]$Port)

try {
    # Use Start-Process so we can capture PID and redirect output.
    $proc = Start-Process `
        -FilePath $uvCmd `
        -ArgumentList $uvArgs `
        -WorkingDirectory $workDir `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $logFile
} catch {
    throw ("Failed to launch server process via uv. Exception: {0}" -f $_.Exception.Message)
}

if ($null -eq $proc -or $proc.Id -le 0) {
    throw "Failed to start the platform UI server process (no PID returned). See log: $logFile"
}

Set-Content -LiteralPath $pidFile -Value ([string]$proc.Id) -Encoding ascii
Set-Content -LiteralPath $portFile -Value ([string]$Port) -Encoding ascii
Write-Host ("Started Platform UI (PID {0})." -f $proc.Id)
Write-Host ("Open http://{0}:{1} (or see server log: {2})" -f $HostAddress, $Port, $logFile)
