"""Clean legacy/unreferenced platform UI assets.

This repo currently contains a legacy JS UI under:
  src/quantdsl_backtest/platform_ui/assets_dist/assets/*.js

and a new React UI served by:
  /static/assets/main.react.js + /static/assets/main.css

Goal of this cleaner:
- Keep only assets that are referenced by assets_dist/index.html and by the new bundle.
- Remove legacy modules (catalog.js, meta.js, inspector.js, etc.) once the React bundle no longer imports them.

This script is intentionally conservative:
- It only deletes an allowlist of known legacy module files.
- It supports --dry-run.

Usage:
  uv run python tools/clean_platform_ui_assets.py --dry-run
  uv run python tools/clean_platform_ui_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist" / "assets"

LEGACY_MODULES = {
    "api.js",
    "catalog.js",
    "download.js",
    "inspector.js",
    "layout.js",
    "meta.js",
    "quality.js",
    "state.js",
    "workspace.js",
}

OTHER_CANDIDATES = {
    # Old build artifacts / maps that are not part of the committed React bundle.
    # Only delete if you confirm they are not referenced by assets_dist/index.html.
    "main.react.js.map",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-vite-junk",
        action="store_true",
        help="Also delete old build artifacts/maps (more risky)",
    )
    args = ap.parse_args()

    if not ASSETS_DIR.exists():
        raise SystemExit(f"Missing assets dir: {ASSETS_DIR}")

    to_delete = set(LEGACY_MODULES)
    if args.include_vite_junk:
        to_delete |= set(OTHER_CANDIDATES)

    existing = {p.name: p for p in ASSETS_DIR.iterdir() if p.is_file()}
    planned = [existing[name] for name in sorted(to_delete) if name in existing]

    if not planned:
        print("Nothing to delete.")
        return

    print("Assets dir:", ASSETS_DIR)
    print("Dry run:" if args.dry_run else "Deleting:")
    for p in planned:
        print("  ", p.name)

    if args.dry_run:
        return

    for p in planned:
        p.unlink()


if __name__ == "__main__":
    main()
