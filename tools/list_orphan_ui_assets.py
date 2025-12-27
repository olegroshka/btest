"""List files under src/quantdsl_backtest/platform_ui/assets_dist/assets that are likely legacy/orphan.

Heuristic:
- Platform serves the UI bundle from assets_dist/assets.
- The current UI is React/Vite-driven and uses /static/assets/main.react.js plus CSS.
- Only files referenced by assets_dist/index.html OR by imports in small JS/CSS files should remain.

Because the main bundle can be large, we avoid fully parsing it; we scan index.html and
smaller assets for /static/assets/<...> references.

Usage:
  uv run python tools/list_orphan_ui_assets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist" / "assets"
INDEX_HTML = ROOT / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist" / "index.html"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    # Ensure consistent utf-8 output even when redirected in Windows shells.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not ASSETS_DIR.exists():
        raise SystemExit(f"Missing assets dir: {ASSETS_DIR}")

    index_txt = read_text(INDEX_HTML) if INDEX_HTML.exists() else ""

    # Collect explicit references from index.html
    referenced = set(re.findall(r"/static/assets/([^\"'\s>]+)", index_txt))

    # Expected entrypoint names (keep these even if index.html changes slightly)
    referenced |= {"main.react.js", "main.css"}

    # Also keep whatever small JS/CSS assets refer to.
    small_assets = [p for p in ASSETS_DIR.glob("*") if p.is_file() and p.stat().st_size < 500_000]
    combined = "\n".join(read_text(p) for p in small_assets if p.suffix in {".css", ".js", ".mjs"})

    for m in re.findall(r"/static/assets/([^\"'\s)]+)", combined):
        referenced.add(m)

    all_files = sorted(p.name for p in ASSETS_DIR.iterdir() if p.is_file())
    orphans = [f for f in all_files if f not in referenced]

    print("Assets dir:", ASSETS_DIR)
    print("Referenced (heuristic):")
    for f in sorted(referenced):
        print("  ", f)

    print("\nCandidate orphan files:")
    for f in orphans:
        print("  ", f)


if __name__ == "__main__":
    main()
