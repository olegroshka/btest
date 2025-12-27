from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> None:
    """Run a command in a way that works on Windows (npm is often npm.cmd).

    On Windows, spawning `npm` directly can fail in some Python environments because
    `npm` is a `.cmd` shim. Using the shell ensures correct resolution.
    """

    if os.name == "nt":
        # Use cmd.exe so `.cmd` shims (npm.cmd) are resolved reliably.
        joined = " ".join(_quote_windows_arg(c) for c in cmd)
        subprocess.check_call(["cmd", "/c", joined], cwd=str(cwd))
        return

    subprocess.check_call(cmd, cwd=str(cwd))


def _quote_windows_arg(s: str) -> str:
    # Minimal quoting suitable for cmd.exe.
    if any(ch in s for ch in [' ', '\t', '"']):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    frontend = repo / "frontend"

    if not frontend.exists():
        raise FileNotFoundError(f"Missing frontend folder: {frontend}")

    pkg = frontend / "package.json"
    if not pkg.exists():
        raise FileNotFoundError(f"Missing package.json: {pkg}")

    # Ensure node_modules exist (common failure mode in fresh clones).
    # We use npm ci when a lockfile is present; otherwise fall back to npm install.
    node_modules = frontend / "node_modules"
    lock = frontend / "package-lock.json"
    if not node_modules.exists():
        if lock.exists():
            _run(["npm", "ci"], cwd=frontend)
        else:
            _run(["npm", "install"], cwd=frontend)

    _run(["npm", "run", "build"], cwd=frontend)

    # Contract: backend serves committed assets from this folder.
    assets_dir = repo / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist" / "assets"
    main_js = assets_dir / "main.react.js"
    main_css = assets_dir / "main.css"
    if not main_js.exists():
        raise FileNotFoundError(
            f"Build did not produce expected bundle: {main_js}. "
            f"Check frontend build output and that Vite is configured to write into assets_dist."
        )
    if not main_css.exists():
        raise FileNotFoundError(
            f"Build did not produce expected css: {main_css}. "
            f"Check frontend build output and that Vite is configured to write into assets_dist."
        )


if __name__ == "__main__":
    main()
