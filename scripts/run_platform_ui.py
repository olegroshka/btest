from __future__ import annotations

"""Run the local Platform UI + API.

Usage (Windows/Powershell):
  uv run python .\scripts\run_platform_ui.py

This is intended as a developer-quality smoke runner for the platform UI.
"""

import uvicorn

from quantdsl_backtest.platform_api.main import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

