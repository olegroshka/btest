from __future__ import annotations

import re
from pathlib import Path


def test_platform_ui_assets_contract_has_entrypoints() -> None:
    """Guards against deleting/moving the UI bundle without updating the server.

    The backend serves a committed SPA build from:
      src/quantdsl_backtest/platform_ui/assets_dist

    Contract:
    - index.html exists
    - its referenced /static/assets/* files exist under assets_dist/assets
    - main.react.js exists (React/Vite UI bundle)
    """

    root = Path(__file__).resolve().parents[3]
    dist_dir = root / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist"
    assets_dir = dist_dir / "assets"

    assert dist_dir.exists(), f"Missing UI dist dir: {dist_dir}"
    assert assets_dir.exists(), f"Missing assets dir: {assets_dir}"

    index_path = dist_dir / "index.html"
    assert index_path.exists(), f"Missing UI index.html: {index_path}"

    index_html = index_path.read_text(encoding="utf-8", errors="ignore")

    # JS bundle is checked into the repo.
    assert (assets_dir / "main.react.js").exists(), "Expected bundled main.react.js"

    # Every /static/assets/<file> referenced from index.html must exist.
    referenced = set(re.findall(r"/static/assets/([^\"'\s>]+)", index_html))
    assert referenced, "Expected index.html to reference at least one /static/assets/* file"

    missing = sorted(f for f in referenced if not (assets_dir / f).exists())
    assert not missing, f"index.html references missing assets: {missing}"
