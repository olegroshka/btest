from __future__ import annotations

# Ensure the project package under `src/` is importable when tests run without an editable install.
# This keeps `pytest` / `uv run pytest` working in fresh environments.
import sys
from pathlib import Path

import pytest


def _repo_root_from_here() -> Path:
    # tests/conftest.py -> repo root
    return Path(__file__).resolve().parent.parent


# Add repo/src to sys.path as early as possible.
# Note: pytest imports conftest before importing tests, so this is safe.
_repo_root = _repo_root_from_here()
_src = str((_repo_root / "src").resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture(autouse=True)
def _prevent_repo_outputs_write(monkeypatch, request):
    """Fail fast if anything tries to write into the repo-level `outputs/` folder.

    Why:
      - Backtest runs in this repo write HTML reports to `outputs/<strategy>/...` by default.
      - Some tests call `run_backtest()` which would otherwise overwrite your real reports.

    Escape hatch:
      - Mark a test with `@pytest.mark.allow_repo_outputs_write` if it *intentionally* writes
        to the repo outputs directory (should be extremely rare; prefer tmp_path).
    """

    if request.node.get_closest_marker("allow_repo_outputs_write") is not None:
        return

    repo_root = _repo_root_from_here()
    outputs_root = (repo_root / "outputs").resolve()

    def _is_under_outputs(p: Path) -> bool:
        try:
            rp = p.resolve()
        except Exception:
            rp = p.absolute()
        try:
            return str(rp).startswith(str(outputs_root))
        except Exception:
            return False

    orig_write_text = Path.write_text
    orig_write_bytes = Path.write_bytes
    orig_open = Path.open

    def guarded_write_text(self: Path, *args, **kwargs):  # type: ignore[override]
        if _is_under_outputs(self):
            raise AssertionError(f"Tests must not write under repo outputs/: {self}")
        return orig_write_text(self, *args, **kwargs)

    def guarded_write_bytes(self: Path, *args, **kwargs):  # type: ignore[override]
        if _is_under_outputs(self):
            raise AssertionError(f"Tests must not write under repo outputs/: {self}")
        return orig_write_bytes(self, *args, **kwargs)

    def guarded_open(self: Path, *args, **kwargs):  # type: ignore[override]
        mode = "r"
        if args:
            mode = args[0]
        else:
            mode = kwargs.get("mode", "r")

        if any(m in str(mode) for m in ("w", "a", "+", "x")) and _is_under_outputs(self):
            raise AssertionError(f"Tests must not write under repo outputs/: {self}")

        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write_text, raising=True)
    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes, raising=True)
    monkeypatch.setattr(Path, "open", guarded_open, raising=True)


def pytest_configure(config: pytest.Config) -> None:
    # Suppress a known external deprecation warning triggered by ArcticDB returning
    # DataFrames backed by unconsolidated pandas managers.
    # We keep this scoped to the exact warning text to avoid hiding real issues.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*BlockManagerUnconsolidated.*",
        category=DeprecationWarning,
    )
