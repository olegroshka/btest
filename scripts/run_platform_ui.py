from __future__ import annotations

"""Run the local Platform UI + API.

Usage (Windows/Powershell):
  uv run python .\scripts\run_platform_ui.py

Optional:
  uv run python .\scripts\run_platform_ui.py --port 8001

This is intended as a developer-quality smoke runner for the platform UI.
"""

import argparse

import uvicorn

from quantdsl_backtest.platform_api.main import app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Platform UI + API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

