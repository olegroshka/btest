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

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root '.platform_ui'
$pidFile = Join-Path $stateDir 'server.pid'

$stopped = @()

# 1) Preferred: PID file shutdown (created by run_platform_ui.ps1)
if (Test-Path $pidFile) {
    $pidText = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($pidText) {
        $serverPid = 0
        [void][int]::TryParse($pidText.Trim(), [ref]$serverPid)
        if ($serverPid -gt 0) {
            $proc = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
            if ($null -ne $proc) {
                Write-Output ("Stopping PID {0} (from {1})..." -f $serverPid, $pidFile)
                Stop-Process -Id $serverPid -Force
                $stopped += $serverPid
            }
        }
    }

    # If we stopped the PID-file process (or it was stale), remove the PID file.
    if (Test-Path $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# 2) Fallback: find by command line
if ($stopped.Count -eq 0) {
    $pattern1 = '*scripts\\run_platform_ui.py*'
    $pattern2 = '*scripts/run_platform_ui.py*'

    $matches = @(Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -like 'python*' -or $_.Name -like 'uv*') -and
            ($_.CommandLine -like $pattern1 -or $_.CommandLine -like $pattern2)
        } |
        Select-Object ProcessId, Name, CommandLine)

    foreach ($m in $matches) {
        Write-Output ("Stopping PID {0} ({1})..." -f $m.ProcessId, $m.Name)
        Stop-Process -Id $m.ProcessId -Force
        $stopped += $m.ProcessId
    }
}

if ($stopped.Count -eq 0) {
    Write-Output 'No running platform UI server process (run_platform_ui.py) found.'
    exit 1
}

Write-Output ("Stopped {0} process(es): {1}" -f $stopped.Count, ($stopped -join ','))
exit 0

