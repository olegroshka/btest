from __future__ import annotations


def test_quality_scan_and_issues_endpoints(monkeypatch):
    """Unit-level route test with a mocked arctic client.

    We validate:
      - POST /api/quality/scan returns scan record
      - GET /api/quality/issues returns rows

    The scan uses catalog_index meta + describe_frame. We mock those layers.
    """

    import pandas as pd
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    # --- fake arctic
    class _Obj:
        def __init__(self, data):
            self.data = data

    class _Lib:
        def __init__(self, df_map):
            self._df_map = df_map

        def read(self, symbol: str):
            return _Obj(self._df_map[symbol])

    class _MetaLib:
        def __init__(self):
            self._store = {}

        def has_symbol(self, symbol: str) -> bool:
            return symbol in self._store

        def read(self, symbol: str):
            return _Obj(self._store[symbol])

        def write(self, symbol: str, data):
            self._store[symbol] = data
            return None

    class _Arctic:
        def __init__(self, meta_lib, data_libs):
            self._meta = meta_lib
            self._data_libs = data_libs

        def get_library(self, name: str, create_if_missing: bool = False):
            if name == "platform_meta/catalog":
                return self._meta
            return self._data_libs[name]

    # meta index contains one symbol
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"px": [1, 2, 3, 4, 5]}, index=idx)

    # create an artificial gap by removing one day
    df_gap = df.drop(df.index[2])

    meta_df = pd.DataFrame(
        [
            {
                "provider": "PARQUET",
                "frequency": "1d",
                "kind": "market_bars",
                "dataset": "ds",
                "entity": "AAA",
                "symbol": "market_bars/ds/AAA",
                "library": "market_data/PARQUET/1d",
                "start": idx.min(),
                "end": idx.max(),
                "updated_at": idx.max(),
                "rows": int(len(df_gap)),
                "cols": 1,
            }
        ]
    )

    meta_lib = _MetaLib()
    # catalog_index is canonical now
    meta_lib.write("catalog_index", meta_df)

    data_libs = {"market_data/PARQUET/1d": _Lib({"market_bars/ds/AAA": df_gap})}
    arctic = _Arctic(meta_lib, data_libs)

    import quantdsl_backtest.platform_api.services.catalog as catalog_svc

    monkeypatch.setattr(catalog_svc, "default_arctic_client", lambda: arctic)

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r_scan = client.post("/api/quality/scan", params={"provider": "PARQUET", "frequency": "1d", "limit": 10})
    assert r_scan.status_code == 200, r_scan.text
    out = r_scan.json()
    assert "scan" in out
    assert out["scan"]["status"] in ("succeeded", "failed")

    r_issues = client.get("/api/quality/issues", params={"provider": "PARQUET"})
    assert r_issues.status_code == 200, r_issues.text
    issues = r_issues.json()
    assert "rows" in issues
    # We should have at least one gap issue
    assert len(issues["rows"]) >= 1

