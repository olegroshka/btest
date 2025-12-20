from __future__ import annotations

import pytest

from quantdsl_backtest.data.resolver import resolve_source


def test_resolve_source_finra_alias():
    assert resolve_source("FINRA:HY_OAS").startswith("fred://")


def test_resolve_source_finra_unknown():
    with pytest.raises(ValueError):
        resolve_source("FINRA:DOES_NOT_EXIST")

