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

param(
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root '.platform_ui'

# Make state/log files port-specific so multiple instances can run concurrently.
$pidFile = Join-Path $stateDir ("server_{0}.pid" -f $Port)
$logFile = Join-Path $stateDir ("server_{0}.log" -f $Port)
$errFile = Join-Path $stateDir ("server_{0}.err.log" -f $Port)
$portFile = Join-Path $stateDir ("server_{0}.port" -f $Port)

# Ensure log files exist so Start-Process redirection always has a target.
foreach ($f in @($logFile, $errFile)) {
    if (!(Test-Path $f)) {
        New-Item -ItemType File -Path $f -Force | Out-Null
    }
}

if (Test-Path $pidFile) {
    $oldPidText = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($oldPidText) {
        $oldPid = 0
        [void][int]::TryParse($oldPidText.Trim(), [ref]$oldPid)
        if ($oldPid -gt 0) {
            $oldProc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($null -ne $oldProc) {
                Write-Output ("Platform UI already running on port {0} (PID {1}). Log: {2}" -f $Port, $oldPid, $logFile)
                exit 0
            }
        }
    }

    # Stale PID file
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host ("Starting Platform UI... host={0} port={1} stdout={2} stderr={3}" -f $HostAddress, $Port, $logFile, $errFile)

function Get-ListeningOwnerPid {
    param([Parameter(Mandatory = $true)][int]$Port)

    # Prefer Get-NetTCPConnection (more structured than parsing netstat)
    try {
        $c = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1)
        if ($c.Count -gt 0 -and $null -ne $c[0]) {
            $pid = [int]$c[0].OwningProcess
            if ($pid -gt 0 -and $pid -ne 4) { return $pid }
        }
    } catch {
        # ignore and fall back
    }

    # Fallback: netstat parsing, but only accept LISTENING lines.
    try {
        $lines = @(netstat -aon | Select-String -Pattern (':'+$Port+'\s') | ForEach-Object { $_.Line })
        foreach ($l in $lines) {
            if ($l -match '\sLISTENING\s+(\d+)\s*$') {
                $pid = 0
                if ([int]::TryParse($Matches[1], [ref]$pid) -and $pid -gt 0 -and $pid -ne 4) {
                    return $pid
                }
            }
        }
    } catch {
        # ignore
    }

    return $null
}

# Fail fast if port is already in use.
$ownerPid = Get-ListeningOwnerPid -Port $Port

if ($null -ne $ownerPid) {
    $procName = $null
    try {
        $p = Get-Process -Id $ownerPid -ErrorAction Stop
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

function Get-ListenerPidForPort {
    param([Parameter(Mandatory = $true)][int]$Port)

    $lines = @()
    try {
        $lines = @(netstat -aon | Select-String -Pattern (':'+$Port+'\s') | ForEach-Object { $_.Line })
    } catch {
        $lines = @()
    }

    foreach ($l in $lines) {
        if ($l -match '\sLISTENING\s+(\d+)\s*$') {
            $listenerPid = 0
            if ([int]::TryParse($Matches[1], [ref]$listenerPid) -and $listenerPid -gt 0) {
                return $listenerPid
            }
        }
    }

    return $null
}

try {
    # Use Start-Process so we can capture PID and redirect output.
    $proc = Start-Process `
        -FilePath $uvCmd `
        -ArgumentList $uvArgs `
        -WorkingDirectory $workDir `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $errFile
} catch {
    throw ("Failed to launch server process via uv. Exception: {0}" -f $_.Exception.Message)
}

if ($null -eq $proc -or $proc.Id -le 0) {
    throw "Failed to start the platform UI server process (no PID returned). See log: $logFile"
}

# Wait briefly for the actual server to bind to the port, then record the listener PID.
$listenerPid = $null
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline) {
    $listenerPid = Get-ListenerPidForPort -Port $Port
    if ($null -ne $listenerPid) { break }
    Start-Sleep -Milliseconds 200
}

# Fall back to launcher PID if we can't detect the listener PID (still useful as a hint).
$pidToRecord = if ($null -ne $listenerPid) { $listenerPid } else { $proc.Id }

Set-Content -LiteralPath $pidFile -Value ([string]$pidToRecord) -Encoding ascii
Set-Content -LiteralPath $portFile -Value ([string]$Port) -Encoding ascii

if ($null -ne $listenerPid) {
    Write-Host ("Started Platform UI (PID {0})." -f $listenerPid)
} else {
    Write-Host ("Started Platform UI (launcher PID {0}; listener PID not detected yet)." -f $proc.Id)
}

Write-Host ("Open http://{0}:{1} (or see server logs: {2} / {3})" -f $HostAddress, $Port, $logFile, $errFile)
