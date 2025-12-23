from __future__ import annotations

import pytest


pytestmark = pytest.mark.slow


def test_platform_api_e2e_catalog_after_running_strategy_parquet_ingest(tmp_path, monkeypatch):
    """E2E: run a real strategy -> ingest parquet bars into Arctic -> exercise API like a user.

    This is a "true" black-box style test:
      1) Run a real example strategy (parquet-backed) with reporting disabled.
      2) Verify the data layer wrote into an isolated ArcticDB LMDB store.
      3) Query the platform API to validate what the UI relies on.

    Isolation:
      - Uses a dedicated LMDB folder under tmp_path via QUANTDSL_ARCTIC_URI.
      - Never touches repo local_cache/.

    Note: This is intentionally slow.
    """

    # Pandas deprecation warning can be triggered by Arctic returning an unconsolidated manager.
    # We don't want to pay the cost of deep copies in production code; suppress here.
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*BlockManagerUnconsolidated.*",
    )

    # 1) Isolated Arctic cache (forward slashes in the URI)
    arctic_root = tmp_path / "arctic_e2e_cache"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # 2) Run example strategy (parquet source), disable reporting outputs
    from quantdsl_backtest.examples.lagging_indecies import build_strategy
    from quantdsl_backtest.engine.backtest_runner import run_backtest

    strat = build_strategy()
    try:
        strat.backtest.reporting.output_dir = None
    except Exception:
        pass

    res = run_backtest(strat)
    assert res is not None

    # Safety: ensure isolated cache exists (LMDB files)
    assert any(arctic_root.rglob("data.mdb")), "Expected Arctic LMDB files under isolated cache"

    # 3) Query API in-process
    from quantdsl_backtest.platform_api.main import create_app
    from fastapi.testclient import TestClient

    c = TestClient(create_app())

    # --- health
    r_health = c.get("/health")
    assert r_health.status_code == 200, r_health.text

    # --- catalog listing
    r = c.get("/api/catalog")
    assert r.status_code == 200, r.text
    payload = r.json()

    assert set(payload.keys()) >= {"libraries"}
    libs = payload.get("libraries")
    assert isinstance(libs, list)

    parquet_libs = [l for l in libs if str(l.get("library", "")).startswith("market_data/PARQUET/")]
    assert parquet_libs, f"Expected PARQUET libraries in catalog, got: {[l.get('library') for l in libs]}"

    # Pick a concrete cached library + symbol.
    # Symbol keys are now per-library: <kind>/<dataset>/<entity>
    library = None
    symbol_in_lib = None

    for lib_rec in parquet_libs:
        lib_name = str(lib_rec.get("library") or "")
        if not lib_name:
            continue
        for s in lib_rec.get("symbols", []) or []:
            sym = s.get("symbol") if isinstance(s, dict) else None
            if isinstance(sym, str) and sym and "/" in sym:
                library = lib_name
                symbol_in_lib = sym
                break
        if library and symbol_in_lib:
            break

    assert isinstance(library, str) and library.startswith("market_data/PARQUET/")
    assert isinstance(symbol_in_lib, str) and symbol_in_lib.count("/") >= 2

    # Ensure the library path isn't accidentally embedding the symbol.
    assert ("market_bars/" not in library)

    # --- meta endpoint: basic shape
    r2 = c.get("/api/catalog/meta")
    assert r2.status_code == 200, r2.text
    meta_all = r2.json()
    assert isinstance(meta_all.get("rows"), list)
    assert int(meta_all.get("count", 0) or 0) == len(meta_all.get("rows", []))

    # --- meta endpoint: filter by provider
    r2p = c.get("/api/catalog/meta", params={"provider": "PARQUET"})
    assert r2p.status_code == 200, r2p.text
    meta = r2p.json()
    assert isinstance(meta.get("rows"), list)
    assert int(meta.get("count", 0) or 0) >= 1
    assert all(r.get("provider") == "PARQUET" for r in meta.get("rows", []))

    # Known invariants for parquet bars in our system
    row0 = meta["rows"][0]
    assert row0.get("provider") == "PARQUET"
    assert row0.get("frequency") == "1d"
    assert row0.get("kind") == "market_bars"
    assert isinstance(row0.get("symbol"), str) and "/" in str(row0.get("symbol"))

    # --- meta endpoint: filter by (provider, frequency)
    r3 = c.get("/api/catalog/meta", params={"provider": "PARQUET", "frequency": "1d"})
    assert r3.status_code == 200, r3.text
    meta_pf = r3.json()
    assert int(meta_pf.get("count", 0) or 0) >= 1
    assert all(r.get("provider") == "PARQUET" for r in meta_pf.get("rows", []))
    assert all(r.get("frequency") == "1d" for r in meta_pf.get("rows", []))

    # Find the meta row for our chosen symbol (by the global symbol key)
    chosen_meta_rows = [r for r in meta_pf.get("rows", []) if r.get("symbol") == symbol_in_lib]
    assert chosen_meta_rows, f"Expected meta for symbol {symbol_in_lib!r} to exist"

    chosen_meta = chosen_meta_rows[0]
    assert chosen_meta.get("dataset")
    assert chosen_meta.get("entity")
    assert chosen_meta.get("start") is not None
    assert chosen_meta.get("end") is not None

    # --- per-symbol metadata (from meta index)
    r4 = c.get(f"/api/catalog/meta/{symbol_in_lib}")
    assert r4.status_code == 200, r4.text
    sym_meta = r4.json()
    assert sym_meta.get("symbol") == symbol_in_lib
    assert isinstance(sym_meta.get("meta"), dict)
    assert sym_meta["meta"].get("provider") == "PARQUET"
    assert sym_meta["meta"].get("frequency") == "1d"

    # --- negative: per-symbol meta of unknown key
    r4n = c.get("/api/catalog/meta/DOES_NOT_EXIST")
    assert r4n.status_code == 200, r4n.text
    assert r4n.json() == {"symbol": "DOES_NOT_EXIST", "meta": None}

    # --- symbol preview (raw frame)
    r5 = c.get(f"/api/catalog/preview/{library}", params={"symbol": symbol_in_lib, "head": 3, "tail": 2})
    assert r5.status_code == 200, r5.text
    prev = r5.json()
    assert prev.get("symbol") == symbol_in_lib
    assert prev.get("library") == library
    assert isinstance(prev.get("head"), list)
    assert isinstance(prev.get("tail"), list)

    # Preview rows must be JSON objects (records)
    if prev.get("head"):
        assert isinstance(prev["head"][0], dict)
    if prev.get("tail"):
        assert isinstance(prev["tail"][0], dict)

    # --- negative: preview invalid args
    r5bad = c.get(f"/api/catalog/preview/{library}", params={"symbol": symbol_in_lib, "head": 2001, "tail": 0})
    assert r5bad.status_code == 400

    # --- negative: preview non-existent library should be a structured 503
    r5missing = c.get(
        "/api/catalog/preview/market_data/PARQUET/1d",
        params={"symbol": "NO_SUCH_SYMBOL", "head": 1, "tail": 0},
    )
    assert r5missing.status_code == 503
    err_missing = r5missing.json()
    assert err_missing.get("error", {}).get("code") == "HTTP_503"

    # --- refresh catalog meta (maintenance endpoint)
    # Should be ok and return stats (even if no-ops)
    r_refresh = c.post("/api/catalog/refresh")
    assert r_refresh.status_code == 200, r_refresh.text
    refresh_payload = r_refresh.json()
    assert refresh_payload.get("status") == "ok"
    assert isinstance(refresh_payload.get("stats"), dict)

    # --- download plan (legacy endpoint is POST)
    source = "parquet://equities/indicies.parquet"

    # Derive entity from the cache key (<kind>/<dataset>/<entity>)
    planned_entity = str(symbol_in_lib).split("/", 2)[-1]
    assert planned_entity

    r6 = c.post(
        "/api/catalog/plan_download",
        params={
            "source": source,
            "kind": "market_bars",
            "frequency": "1d",
            "start": "2015-01-01",
            "end": "2015-02-01",
            "entities": planned_entity,
        },
    )
    assert r6.status_code == 200, r6.text
    plan_payload = r6.json()
    plan = plan_payload.get("plan")
    assert isinstance(plan, list)
    assert len(plan) >= 1

    first = plan[0]
    assert isinstance(first, dict)
    entity = first.get("entity")
    assert isinstance(entity, str) and entity

    # --- download raw data (legacy endpoint is POST)
    r7 = c.post(
        "/api/catalog/download",
        params={
            "source": source,
            "kind": "market_bars",
            "frequency": "1d",
            "start": "2015-01-01",
            "end": "2015-02-01",
            "entities": entity,
            "dry_run": False,
        },
    )
    assert r7.status_code == 200, r7.text
    dl = r7.json()
    assert dl.get("dry_run") is False
    assert dl.get("kind") == "market_bars"
    assert dl.get("frequency") == "1d"
    assert isinstance(dl.get("entities"), list)
    assert entity in dl.get("entities", []), f"Expected downloaded entities to include {entity!r}"

    # If provider stats are available, ensure structure is sane
    if dl.get("cache_stats") is not None:
        assert isinstance(dl["cache_stats"], dict)

    if dl.get("stats_by_entity") is not None:
        assert isinstance(dl["stats_by_entity"], dict)

    if dl.get("actions_by_entity") is not None:
        assert isinstance(dl["actions_by_entity"], dict)

    # --- download plan v2 (body-based) should behave similarly
    # NOTE: v2 endpoint removed intentionally; /plan_download is the single supported API.

    # --- download v2: dry run
    r7v2 = c.post(
        "/api/catalog/download",
        json={
            "source": source,
            "kind": "market_bars",
            "frequency": "1d",
            "start": "2015-01-01",
            "end": "2015-02-01",
            "entities": [entity],
            "dry_run": True,
        },
    )
    assert r7v2.status_code == 200, r7v2.text
    dl2 = r7v2.json()
    assert dl2.get("dry_run") is True
    assert isinstance(dl2.get("request"), dict)
    assert dl2.get("plan") is None or isinstance(dl2.get("plan"), list)
