from __future__ import annotations

import warnings

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.anyio]


async def test_platform_ui_httpx_e2e_smoke(tmp_path, monkeypatch):
    """E2E-ish: use an isolated Arctic store + in-process ASGI client (httpx).

    Scope (what this protects):
      - The Platform UI HTML page renders and contains the controls we expect.
      - The key API endpoints used by the UI work end-to-end on a real cache:
          /api/catalog
          /api/catalog/meta
          /api/catalog/meta/{symbol}
          /api/catalog/preview/{library}?symbol=...
          /api/catalog/download  (dry-run)

    Notes:
      - We don't execute browser JS here; we validate the backend contract the UI relies on.
      - This test is marked slow because it runs an example strategy to populate the cache.
      - Uses a dedicated LMDB path under tmp_path; does not touch repo local_cache/.
    """

    # Pandas deprecation warning can be triggered by Arctic returning an unconsolidated manager.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*BlockManagerUnconsolidated.*",
    )

    # Isolated Arctic cache
    arctic_root = tmp_path / "arctic_ui_httpx_e2e"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # Populate cache via a real parquet-backed example strategy
    from quantdsl_backtest.examples.lagging_indecies import build_strategy
    from quantdsl_backtest.engine.backtest_runner import run_backtest

    strat = build_strategy()
    try:
        strat.backtest.reporting.output_dir = None
    except Exception:
        pass

    res = run_backtest(strat)
    assert res is not None
    assert any(arctic_root.rglob("data.mdb")), "Expected Arctic LMDB files under isolated cache"

    # FRED downloads must not hit the network during tests.
    monkeypatch.setenv("FRED_API_KEY", "DUMMY")

    import pandas as pd
    import quantdsl_backtest.data.market as market_mod

    def _fake_fetch_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
        return pd.DataFrame({"value": range(len(idx))}, index=idx)

    monkeypatch.setattr(market_mod, "fetch_fred_series", _fake_fetch_fred_series)

    # In-process ASGI app + httpx client
    from quantdsl_backtest.platform_api.main import create_app

    import httpx

    app = create_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        # --- UI loads
        r_ui = await c.get("/")
        assert r_ui.status_code == 200
        assert "text/html" in (r_ui.headers.get("content-type") or "")
        html = r_ui.text
        assert "Platform UI" in html

        # UI shell is now React/Vite-driven; the legacy controls are no longer server-rendered.
        # Protect the important invariants:
        #   - We served the UI shell
        #   - It loads a JS bundle
        assert (
            ("/static/assets/main.mjs" in html)
            or ("/static/assets/main.js" in html)
            or ("/static/assets/main.react.js" in html)
        )

        # Keep this guard: plotly is still available for the inspector.
        assert "plotly" in html.lower()

        # (Legacy IDs like catalogSearch/btnDryRun are now created by JS at runtime.)

        # --- API sanity checks continue below

        # Quality endpoints must work (UI renders these controls client-side)

        # Trigger a quality scan for PARQUET 1d
        r_qscan = await c.post("/api/quality/scan", params={"provider": "PARQUET", "frequency": "1d", "limit": 20})
        assert r_qscan.status_code == 200, r_qscan.text
        qscan = r_qscan.json()
        assert "scan" in qscan

        # Issues endpoint should respond
        r_qissues = await c.get("/api/quality/issues", params={"provider": "PARQUET", "frequency": "1d"})
        assert r_qissues.status_code == 200, r_qissues.text
        qissues = r_qissues.json()
        assert isinstance(qissues.get("rows"), list)

        # --- catalog
        r_cat = await c.get("/api/catalog")
        assert r_cat.status_code == 200, r_cat.text
        cat = r_cat.json()
        assert isinstance(cat.get("libraries"), list)

        # Ensure meta index is up-to-date for this isolated cache.
        r_refresh = await c.post("/api/catalog/refresh")
        assert r_refresh.status_code == 200, r_refresh.text

        # choose a parquet library and a symbol
        parquet_libs = [l for l in cat["libraries"] if str(l.get("library", "")).startswith("market_data/PARQUET/")]
        assert parquet_libs, f"No parquet libs found in: {[l.get('library') for l in cat['libraries']]}"

        library = None
        symbol = None
        for lib_rec in parquet_libs:
            lib_name = str(lib_rec.get("library") or "")
            for s in lib_rec.get("symbols", []) or []:
                if isinstance(s, dict):
                    sym = s.get("symbol")
                else:
                    sym = None
                if isinstance(sym, str) and sym and sym.count("/") >= 2:
                    library = lib_name
                    symbol = sym
                    break
            if library and symbol:
                break

        assert library and symbol

        # --- meta listing
        # Don't over-assume provider availability; just ensure the meta index exists and is queryable.
        r_meta_all = await c.get("/api/catalog/meta", params={"limit": 500})
        assert r_meta_all.status_code == 200, r_meta_all.text
        meta_all = r_meta_all.json()
        assert int(meta_all.get("count", 0) or 0) == len(meta_all.get("rows", []))
        assert meta_all.get("rows"), "Expected non-empty meta index after /api/catalog/refresh"

        # Prefer a meta row that belongs to the selected library so /preview works.
        meta_rows = [r for r in (meta_all.get("rows") or []) if str((r or {}).get("library") or "") == str(library)]
        if not meta_rows:
            # Fallback: take any row and let preview validate the backend can handle it.
            meta_rows = list(meta_all.get("rows") or [])

        meta_row = meta_rows[0]
        meta_symbol = str((meta_row or {}).get("symbol") or "")
        assert meta_symbol

        # Preview should work for a meta symbol.
        r_prev_meta = await c.get(f"/api/catalog/preview/{library}", params={"symbol": meta_symbol, "head": 1, "tail": 1})
        assert r_prev_meta.status_code == 200, r_prev_meta.text

        # Library filter should also work (even if historical meta rows have library null).
        r_meta_lib = await c.get("/api/catalog/meta", params={"library": library, "limit": 5})
        assert r_meta_lib.status_code == 200, r_meta_lib.text
        meta_lib = r_meta_lib.json()
        assert isinstance(meta_lib.get("rows"), list)

        # --- per-symbol meta
        r_sym_meta = await c.get(f"/api/catalog/meta/{symbol}")
        assert r_sym_meta.status_code == 200, r_sym_meta.text
        sym_meta = r_sym_meta.json()
        assert sym_meta.get("symbol") == symbol
        assert isinstance(sym_meta.get("meta"), dict)
        meta_dict = sym_meta["meta"]
        assert meta_dict.get("provider") == "PARQUET"

        # Meta invariants we will use for meta-driven download payload
        assert meta_dict.get("kind") == "market_bars"
        assert meta_dict.get("frequency") == "1d"
        assert meta_dict.get("entity")  # should exist for parquet market bars

        # --- preview
        r_prev = await c.get(f"/api/catalog/preview/{library}", params={"symbol": symbol, "head": 5, "tail": 5})
        assert r_prev.status_code == 200, r_prev.text
        prev = r_prev.json()
        assert prev.get("library") == library
        assert prev.get("symbol") == symbol
        assert isinstance(prev.get("head"), list)
        assert isinstance(prev.get("tail"), list)

        # --- dry-run download (body-based)
        source = "parquet://equities/indicies.parquet"

        # IMPORTANT: emulate the UI's meta-driven payload
        # Prefer entity from meta; fallback is derived from the symbol key
        entity = str(meta_dict.get("entity") or str(symbol).split("/", 2)[-1])
        assert entity

        r_dl = await c.post(
            "/api/catalog/download",
            json={
                "source": source,
                "kind": str(meta_dict.get("kind")),
                "frequency": str(meta_dict.get("frequency")),
                "start": "2015-01-01",
                "end": "2015-02-01",
                "entities": [entity],
                "dry_run": True,
            },
        )
        assert r_dl.status_code == 200, r_dl.text
        dl = r_dl.json()
        assert dl.get("dry_run") is True
        assert isinstance(dl.get("request"), dict)
        assert dl["request"].get("kind") == str(meta_dict.get("kind"))
        assert dl["request"].get("frequency") == str(meta_dict.get("frequency"))

        # The dry-run endpoint should echo the requested entity either directly or via plan.
        # Some providers may not populate top-level entities consistently.
        entities_echo = dl.get("entities")
        if entities_echo is not None:
            assert entity in (entities_echo or [])
        else:
            plan = dl.get("plan")
            assert plan is not None, "Expected either top-level entities or a plan in dry-run response"
            # plan items are dicts; ensure our entity appears in at least one item
            assert any((p or {}).get("entity") == entity for p in (plan or []))

        # =====================================================================================
        # FRED: ingest via download endpoint, then validate catalog/meta/preview
        # =====================================================================================
        fred_source = "fred://CPIAUCSL"

        r_fred_dl = await c.post(
            "/api/catalog/download",
            json={
                "source": fred_source,
                "kind": "market_bars",
                "frequency": "1d",
                "start": "2024-01-01",
                "end": "2024-01-10",
                "entities": ["CPIAUCSL"],
                "dry_run": False,
            },
        )
        assert r_fred_dl.status_code == 200, r_fred_dl.text
        fred_dl = r_fred_dl.json()
        assert fred_dl.get("dry_run") is False
        assert fred_dl.get("frequency") == "1d"

        # Refresh catalog and locate CPIAUCSL (don't assume library naming/prefix)
        r_cat2 = await c.get("/api/catalog")
        assert r_cat2.status_code == 200, r_cat2.text
        cat2 = r_cat2.json()

        fred_library = None
        fred_symbol = None
        for lib_rec in cat2.get("libraries", []) or []:
            lib_name = str(lib_rec.get("library") or "")
            for s in lib_rec.get("symbols", []) or []:
                sym = s.get("symbol") if isinstance(s, dict) else None
                if isinstance(sym, str) and "CPIAUCSL" in sym:
                    fred_library = lib_name
                    fred_symbol = sym
                    break
            if fred_library:
                break

        assert fred_library and fred_symbol, (
            "Expected CPIAUCSL to appear in catalog after FRED download; "
            f"libs={[l.get('library') for l in cat2.get('libraries', [])]}"
        )

        # Meta listing should include FRED rows, and our chosen symbol should be there.
        r_fred_meta_list = await c.get("/api/catalog/meta", params={"provider": "FRED", "frequency": "1d"})
        assert r_fred_meta_list.status_code == 200, r_fred_meta_list.text
        fred_meta_list = r_fred_meta_list.json()
        assert any(r.get("symbol") == fred_symbol for r in fred_meta_list.get("rows", [])), (
            f"Expected {fred_symbol} in FRED meta index"
        )

        # Per-symbol meta should reflect provider + entity
        r_fred_meta = await c.get(f"/api/catalog/meta/{fred_symbol}")
        assert r_fred_meta.status_code == 200, r_fred_meta.text
        fred_meta = r_fred_meta.json()
        assert fred_meta.get("symbol") == fred_symbol
        assert isinstance(fred_meta.get("meta"), dict)
        assert str(fred_meta["meta"].get("provider")).upper() == "FRED"
        assert fred_meta["meta"].get("entity") == "CPIAUCSL"
        assert fred_meta["meta"].get("frequency") == "1d"

        # --- describe (accurate analytics)
        r_desc = await c.get(f"/api/catalog/describe/{library}", params={"symbol": symbol})
        assert r_desc.status_code == 200, r_desc.text
        desc = r_desc.json()
        assert desc.get("library") == library
        assert desc.get("symbol") == symbol
        assert isinstance(desc.get("rows"), int)
        assert isinstance(desc.get("columns"), list)
        assert isinstance(desc.get("dtypes"), dict)
        assert isinstance(desc.get("missing"), dict)
        assert isinstance(desc.get("gaps"), dict)
        assert "missing_timestamps_sample" in desc["gaps"]
        assert "duplicate_timestamps" in desc["gaps"]
        assert isinstance(desc["gaps"]["missing_timestamps_sample"], list)
        assert len(desc["gaps"]["missing_timestamps_sample"]) <= 10
        assert int(desc["gaps"]["duplicate_timestamps"]) >= 0
        assert "missing_intervals_sample" in desc["gaps"]
        assert "duplicate_timestamps_sample" in desc["gaps"]
        assert "max_gap_periods" in desc["gaps"]
        assert "max_gap_days" in desc["gaps"]
        assert isinstance(desc["gaps"]["missing_intervals_sample"], list)
        assert len(desc["gaps"]["missing_intervals_sample"]) <= 5
        assert isinstance(desc["gaps"]["duplicate_timestamps_sample"], list)
        assert len(desc["gaps"]["duplicate_timestamps_sample"]) <= 10
        assert int(desc["gaps"]["max_gap_periods"]) >= 0
        assert int(desc["gaps"]["max_gap_days"]) >= 0

        # Preview should return head/tail
        r_fred_prev = await c.get(f"/api/catalog/preview/{fred_library}", params={"symbol": fred_symbol, "head": 3, "tail": 3})
        assert r_fred_prev.status_code == 200, r_fred_prev.text
        fred_prev = r_fred_prev.json()
        assert fred_prev.get("library") == fred_library
        assert fred_prev.get("symbol") == fred_symbol
        assert isinstance(fred_prev.get("head"), list)
        assert isinstance(fred_prev.get("tail"), list)

        # FRED describe
        r_fred_desc = await c.get(f"/api/catalog/describe/{fred_library}", params={"symbol": fred_symbol})
        assert r_fred_desc.status_code == 200, r_fred_desc.text
        fred_desc = r_fred_desc.json()
        assert fred_desc.get("library") == fred_library
        assert fred_desc.get("symbol") == fred_symbol
        assert isinstance(fred_desc.get("rows"), int)
        assert fred_desc.get("rows") > 0
        assert isinstance(fred_desc.get("columns"), list)
        assert isinstance(fred_desc.get("missing"), dict)
        assert isinstance(fred_desc.get("gaps"), dict)
        assert "missing_timestamps_sample" in fred_desc["gaps"]
        assert "duplicate_timestamps" in fred_desc["gaps"]
        assert isinstance(fred_desc["gaps"]["missing_timestamps_sample"], list)
        assert len(fred_desc["gaps"]["missing_timestamps_sample"]) <= 10
        assert int(fred_desc["gaps"]["duplicate_timestamps"]) >= 0
        assert "missing_intervals_sample" in fred_desc["gaps"]
        assert "duplicate_timestamps_sample" in fred_desc["gaps"]
        assert "max_gap_periods" in fred_desc["gaps"]
        assert "max_gap_days" in fred_desc["gaps"]
        assert isinstance(fred_desc["gaps"]["missing_intervals_sample"], list)
        assert isinstance(fred_desc["gaps"]["duplicate_timestamps_sample"], list)
        assert int(fred_desc["gaps"]["max_gap_periods"]) == 0
        assert int(fred_desc["gaps"]["max_gap_days"]) == 0
        assert fred_desc["gaps"]["missing_intervals_sample"] == []
        assert fred_desc["gaps"]["duplicate_timestamps_sample"] == []

        # Dry-run: meta-driven payload invariants for FRED too
        r_fred_dry = await c.post(
            "/api/catalog/download",
            json={
                "source": fred_source,
                "kind": str(fred_meta["meta"].get("kind")),
                "frequency": str(fred_meta["meta"].get("frequency")),
                "start": "2024-01-01",
                "end": "2024-01-10",
                "entities": ["CPIAUCSL"],
                "dry_run": True,
            },
        )
        assert r_fred_dry.status_code == 200, r_fred_dry.text
        fred_dry = r_fred_dry.json()
        assert fred_dry.get("dry_run") is True
        assert isinstance(fred_dry.get("request"), dict)
        assert fred_dry["request"].get("kind") == str(fred_meta["meta"].get("kind"))
        assert fred_dry["request"].get("frequency") == str(fred_meta["meta"].get("frequency"))
