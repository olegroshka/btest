param(
  [string]$PytestArgs = "-q",
  [string]$OutFile = "local_cache/test_logs/pytest_latest.txt"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

$of = Join-Path $root $OutFile
$od = Split-Path -Parent $of
if (-not (Test-Path $od)) { New-Item -ItemType Directory -Force -Path $od | Out-Null }

# Run pytest, tee output to console + file (preserves exit code)
$cmd = "pytest $PytestArgs"
Write-Host "[run_tests_capture] $cmd" -ForegroundColor Cyan

& pytest $PytestArgs 2>&1 | Tee-Object -FilePath $of
$exit = $LASTEXITCODE

Write-Host "[run_tests_capture] output saved to: $of" -ForegroundColor Cyan

exit $exit

