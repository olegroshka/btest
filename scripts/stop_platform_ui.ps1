<#
Stops the running QuantDSL Backtest Platform UI server.

This script stops the server using either:
- a PID file created by `scripts\run_platform_ui.ps1` (preferred), or
- a process search for a command line containing `scripts\run_platform_ui.py`.

Usage:
  PowerShell:
    .\scripts\stop_platform_ui.ps1

Exit codes:
  0 - stopped at least one matching process
  1 - no matching process found
#>

param(
    [int]$Port = 0
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root '.platform_ui'

# If a port was specified, use per-port state files; otherwise fall back to legacy files.
$pidFile = if ($Port -gt 0) { Join-Path $stateDir ("server_{0}.pid" -f $Port) } else { Join-Path $stateDir 'server.pid' }
$portFile = if ($Port -gt 0) { Join-Path $stateDir ("server_{0}.port" -f $Port) } else { Join-Path $stateDir 'server.port' }

$stopped = @()

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId
    )

    # Stop children first (best-effort), then the parent.
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessId)
    foreach ($c in $children) {
        if ($c -and ($c -ne $ProcessId)) {
            Stop-ProcessTree -ProcessId ([int]$c)
        }
    }

    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-Output ("Stopping PID {0} ({1})..." -f $ProcessId, $proc.ProcessName)
        try {
            Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        } catch {
            # race: already exited
        }
        $script:stopped += $ProcessId
    }
}

function Stop-ByPort {
    param(
        [Parameter(Mandatory = $true)][int]$Port
    )

    # Find the listening owner PID(s) for this port.
    $lines = @()
    try {
        # netstat is available everywhere and includes PID.
        $lines = @(netstat -aon | Select-String -Pattern (':'+$Port+'\s') | ForEach-Object { $_.Line })
    } catch {
        $lines = @()
    }

    $pids = New-Object System.Collections.Generic.HashSet[int]
    foreach ($l in $lines) {
        if ($l -match '\sLISTENING\s+(\d+)\s*$') {
            $listenerPid = [int]$Matches[1]
            [void]$pids.Add($listenerPid)
        }
    }

    foreach ($procId in $pids) {
        if ($procId -gt 0 -and -not ($script:stopped -contains $procId)) {
            Stop-ProcessTree -ProcessId $procId
        }
    }
}

function Stop-AllKnownPorts {
    # Discover ports from per-port state files: server_<port>.port
    $ports = New-Object System.Collections.Generic.HashSet[int]

    try {
        $files = @(Get-ChildItem -LiteralPath $stateDir -Filter 'server_*.port' -File -ErrorAction SilentlyContinue)
        foreach ($f in $files) {
            $txt = (Get-Content -LiteralPath $f.FullName -ErrorAction SilentlyContinue | Select-Object -First 1)
            $p = 0
            if ($txt -and [int]::TryParse($txt.Trim(), [ref]$p) -and $p -gt 0) {
                [void]$ports.Add($p)
            }
        }
    } catch {
        # ignore
    }

    # Also include legacy port file if present.
    $legacyPortFile = Join-Path $stateDir 'server.port'
    if (Test-Path $legacyPortFile) {
        $txt = (Get-Content -LiteralPath $legacyPortFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        $p = 0
        if ($txt -and [int]::TryParse($txt.Trim(), [ref]$p) -and $p -gt 0) {
            [void]$ports.Add($p)
        }
    }

    foreach ($p in $ports) {
        Stop-ByPort -Port $p

        # Also attempt PID-file kill for this port (best-effort)
        $pf = Join-Path $stateDir ("server_{0}.pid" -f $p)
        if (Test-Path $pf) {
            $pidText = (Get-Content -LiteralPath $pf -ErrorAction SilentlyContinue | Select-Object -First 1)
            $serverPid = 0
            if ($pidText -and [int]::TryParse($pidText.Trim(), [ref]$serverPid) -and $serverPid -gt 0) {
                if (-not ($script:stopped -contains $serverPid)) {
                    Stop-ProcessTree -ProcessId $serverPid
                }
            }
            Remove-Item -LiteralPath $pf -Force -ErrorAction SilentlyContinue
        }

        $portState = Join-Path $stateDir ("server_{0}.port" -f $p)
        if (Test-Path $portState) { Remove-Item -LiteralPath $portState -Force -ErrorAction SilentlyContinue }
    }
}

# 0) Most reliable: stop by port (if we know it)
if ($Port -le 0) {
    # If no explicit port was provided, stop ALL known instances (per-port state files),
    # then do a short retry pass to catch any late-spawned children.
    Stop-AllKnownPorts

    # Retry loop (handles uv spawning and brief shutdown delays)
    for ($i = 0; $i -lt 3; $i++) {
        $before = $stopped.Count
        Stop-AllKnownPorts
        if ($stopped.Count -eq $before) { break }
        Start-Sleep -Milliseconds 400
    }

    # Keep old behavior as a final fallback on 8000 if no state was present.
    if ($stopped.Count -eq 0) {
        $port = 8000
        if (Test-Path $portFile) {
            $portText = (Get-Content -LiteralPath $portFile -ErrorAction SilentlyContinue | Select-Object -First 1)
            if ($portText) {
                $p = 0
                [void][int]::TryParse($portText.Trim(), [ref]$p)
                if ($p -gt 0) { $port = $p }
            }
        }
        Stop-ByPort -Port $port
    }
} else {
    $port = $Port
    Stop-ByPort -Port $port
}

# 1) PID file shutdown (created by run_platform_ui.ps1)
if (Test-Path $pidFile) {
    $pidText = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($pidText) {
        $serverPid = 0
        [void][int]::TryParse($pidText.Trim(), [ref]$serverPid)
        if ($serverPid -gt 0 -and -not ($stopped -contains $serverPid)) {
            Stop-ProcessTree -ProcessId $serverPid
        }
    }

    # Remove PID file (even if stale)
    if (Test-Path $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# 2) Fallback: find by command line
# When stopping all instances, we can still have orphan processes with no state files.
if ($stopped.Count -eq 0 -or $Port -le 0) {
    $pattern1 = '*scripts\\run_platform_ui.py*'
    $pattern2 = '*scripts/run_platform_ui.py*'

    $matches = @(Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -like 'python*' -or $_.Name -like 'uv*') -and
            ($_.CommandLine -like $pattern1 -or $_.CommandLine -like $pattern2)
        } |
        Select-Object ProcessId, Name, CommandLine)

    foreach ($m in $matches) {
        Stop-ProcessTree -ProcessId ([int]$m.ProcessId)
    }
}

# Remove port file too (harmless if missing)
if (Test-Path $portFile) {
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
}

# Also remove legacy state files if we stopped something and caller used -Port
if ($Port -gt 0) {
    $legacyPid = Join-Path $stateDir 'server.pid'
    $legacyPort = Join-Path $stateDir 'server.port'
    if (Test-Path $legacyPid) { Remove-Item -LiteralPath $legacyPid -Force -ErrorAction SilentlyContinue }
    if (Test-Path $legacyPort) { Remove-Item -LiteralPath $legacyPort -Force -ErrorAction SilentlyContinue }
}

if ($stopped.Count -eq 0) {
    Write-Output ("No running platform UI server found (checked port {0}, PID file, and run_platform_ui.py command line)." -f $port)
    exit 1
}

$uniqueStopped = @($stopped | Sort-Object -Unique)
Write-Output ("Stopped {0} process(es): {1}" -f $uniqueStopped.Count, ($uniqueStopped -join ','))
exit 0

