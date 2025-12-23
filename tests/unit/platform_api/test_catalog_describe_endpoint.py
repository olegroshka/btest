from __future__ import annotations


def test_catalog_describe_endpoint_smoke(monkeypatch, tmp_path):
    """Smoke test for /api/catalog/describe/{library}.

    We don't need a real ArcticDB here: we mock the arctic client + library.read.
    """

    import pandas as pd
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    # Fake Arctic library
    class _Obj:
        def __init__(self, data):
            self.data = data

    class _Lib:
        def __init__(self, df):
            self._df = df

        def read(self, symbol: str):
            return _Obj(self._df)

    class _Arctic:
        def __init__(self, lib):
            self._lib = lib

        def get_library(self, name: str, create_if_missing: bool = False):
            return self._lib

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0], "b": [10, 11, 12, 13, 14]}, index=idx)

    fake_arctic = _Arctic(_Lib(df))

    import quantdsl_backtest.platform_api.services.catalog as catalog_svc

    monkeypatch.setattr(catalog_svc, "default_arctic_client", lambda: fake_arctic)

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get(
        "/api/catalog/describe/market_data/ANY/1d",
        params={"symbol": "market_bars/x/y"},
    )
    assert r.status_code == 200, r.text
    out = r.json()

    assert out["library"] == "market_data/ANY/1d"
    assert out["symbol"] == "market_bars/x/y"
    assert out["rows"] == 5
    assert "a" in out["columns"] and "b" in out["columns"]

    assert out["index_start"].startswith("2024-01-01")
    assert out["index_end"].startswith("2024-01-05")

    assert out["missing"]["a"] == 1
    assert out["missing"]["b"] == 0

    # numeric stats exist for numeric columns
    assert "a" in out["numeric"]
    assert "b" in out["numeric"]
    assert out["numeric"]["a"]["min"] == 1.0
    assert out["numeric"]["b"]["max"] == 14.0

    assert "non_null_pct" in out
    assert "unique" in out
    assert "gaps" in out

    # non-null pct should reflect missing
    assert out["non_null_pct"]["a"] == 0.8
    assert out["unique"]["b"] == 5

    # daily index should infer a daily-ish frequency and no gaps
    assert out["gaps"]["missing_periods"] == 0
    assert "missing_timestamps_sample" in out["gaps"]
    assert "duplicate_timestamps" in out["gaps"]
    assert out["gaps"]["duplicate_timestamps"] == 0
    assert out["gaps"]["missing_timestamps_sample"] == []

    assert "missing_intervals_sample" in out["gaps"]
    assert "duplicate_timestamps_sample" in out["gaps"]
    assert "max_gap_periods" in out["gaps"]
    assert "max_gap_days" in out["gaps"]

    assert out["gaps"]["missing_intervals_sample"] == []
    assert out["gaps"]["duplicate_timestamps_sample"] == []
    assert out["gaps"]["max_gap_periods"] == 0
    assert out["gaps"]["max_gap_days"] == 0
