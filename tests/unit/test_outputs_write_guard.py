from __future__ import annotations

from pathlib import Path

import pytest


def test_guard_blocks_repo_outputs_write():
    # This relies on the autouse fixture in tests/conftest.py.
    with pytest.raises(AssertionError):
        (Path("outputs") / "_should_not_exist.html").write_text("nope", encoding="utf-8")

