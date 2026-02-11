# Quick activation script for btest project
# Usage: .\activate.ps1

Set-Location "E:\Personal\Business & Investments\Trading portfolio\Cogilator\btest"
& .\.venv\Scripts\Activate.ps1

Write-Host "✓ Virtual environment activated (Python 3.11.14)" -ForegroundColor Green
Write-Host "✓ Working directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Cyan
Write-Host "  uv run python <script>     - Run script with uv"
Write-Host "  python <script>            - Run script with activated env"
Write-Host "  uv sync                    - Sync dependencies"
Write-Host "  deactivate                 - Exit virtual environment"
