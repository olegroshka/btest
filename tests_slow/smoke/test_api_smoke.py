from __future__ import annotations

import os
import pathlib
import subprocess
import time

import httpx
import pytest


def _read_base_url() -> str:
    root = pathlib.Path(__file__).resolve().parents[2]
    port_file = root / ".platform_ui" / "server.port"
    port = 8000
    try:
        if port_file.exists():
            port = int(port_file.read_text(encoding="utf-8").strip())
    except Exception:
        port = 8000
    return f"http://127.0.0.1:{port}/api"


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _ensure_server_running(base_url: str) -> None:
    """Ensure a local server is reachable. Optionally auto-start it for dev flows."""

    health = base_url.replace("/api", "/health")

    def _up() -> bool:
        try:
            r = httpx.get(health, timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    if _up():
        return

    # Dev-friendly mode: start server automatically for smoke tests.
    if _env_flag("SMOKE_AUTOSTART_SERVER", False):
        root = pathlib.Path(__file__).resolve().parents[2]
        # Use the PowerShell wrapper if available on Windows; it's already PID/log aware.
        ps1 = root / "scripts" / "run_platform_ui.ps1"
        py = root / "scripts" / "run_platform_ui.py"

        if os.name == "nt" and ps1.exists():
            subprocess.check_call(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], cwd=str(root))
        else:
            subprocess.check_call(["uv", "run", "python", str(py)], cwd=str(root))

        # Wait briefly for boot.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if _up():
                return
            time.sleep(0.1)

    # Still down -> fail with actionable message.
    raise AssertionError(
        f"Smoke tests require a running server. Could not reach {base_url}. "
        f"Start it with: uv run python scripts/run_platform_ui.py (or scripts/run_platform_ui.ps1). "
        f"Or set SMOKE_AUTOSTART_SERVER=1 to auto-start it during smoke runs."
    )


@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.manual
def test_api_filters_smoke():
    """Manual smoke test: check API filters on a live server.

    NOTE: base_url is resolved from .platform_ui/server.port when present.
    """

    base_url = _read_base_url()
    _ensure_server_running(base_url)

    check_meta(base_url)


def check_meta(base_url: str) -> None:
    print('--- Fetching all meta ---')
    r = httpx.get(f"{base_url}/catalog/meta")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get('rows'), list)

    rows = data.get("rows", [])
    print(f"Total rows: {len(rows)}")

    if not rows:
        print("Empty catalog index!")
        return

    first = rows[0]
    lib = first.get("library")
    sym = first.get("symbol")
    ent = first.get("entity")
    print(f"First row: lib={lib}, sym={sym}, ent={ent}")

    print(f"--- Querying specifically for lib={lib}, sym={sym} ---")
    params = {"library": lib, "symbol": sym}
    r2 = httpx.get(f"{base_url}/catalog/meta", params=params)
    data2 = r2.json()
    rows2 = data2.get("rows", [])
    print(f"Filtered rows: {len(rows2)}")
    if rows2:
        match = rows2[0].get('symbol') == sym
        print(f"Match: {match}")
        assert match, f"Expected symbol {sym}, got {rows2[0].get('symbol')}"
    else:
        print("FAILED TO FIND BY LIB/SYM")
        assert False, "Failed to find row by library and symbol"
