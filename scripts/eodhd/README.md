# EODHD Workflow Runbook

This directory contains the operational fetchers and factual manifests for the `btest`-owned EODHD data lanes.

## What lives where

- `README.md` (this file): operator-facing entry point for how to run, resume, and audit the EODHD fetchers.
- `EODHD_*_MANIFEST.md`: per-lane factual inventories of scope, local artefacts, and current observed counts.
- `fetch_eodhd_*.py`: the actual fetchers.
- `tmp_poll_eodhd_progress.py` at repo root: ad hoc progress snapshot across the ETF and index-reference lanes.

## Current EODHD lane map

### US

- Common-stock lane: `EODHD_US_COMMON_STOCK_DATA_MANIFEST.md`
- ETF lane: `EODHD_US_ETF_DATA_MANIFEST.md`
- Index / benchmark lane: `EODHD_INDEX_REF_DATA_MANIFEST.md`

### UK/EU

- Common-stock lane: `EODHD_UK_EU_DATA_MANIFEST.md`
- ETF lane: `EODHD_UK_EU_ETF_DATA_MANIFEST.md`
- Index / benchmark lane: `EODHD_UK_EU_INDEX_REF_DATA_MANIFEST.md`

## Resume model

The fetchers are designed to be rerun normally.

Continuation state is persisted in the per-lane sidecars:

- prices: `prices_fetch_state.csv`
- dividends: `dividends_fetch_state.csv` plus `dividends_fetch_audit.csv`
- splits: `splits_fetch_state.csv` plus `splits_fetch_audit.csv`

Normal reruns should **not** use `--full-refresh`.

On a normal rerun, the scripts will:

- skip pairs already covered through the requested `--to` bound,
- continue incremental tails for pairs with local history,
- preserve explicit `empty` states for tickers where the provider returned no dividend/split history,
- and merge new output with existing parquet files.

Use `--full-refresh` only when you intentionally want to rebuild an output from scratch.

## Typical run order

Refresh universes first, then prices, then event histories.

### US ETF

```powershell
Set-Location "C:\Users\olegr\PycharmProjects\btest"
uv run python scripts/eodhd/fetch_eodhd_us_etf_universe.py
uv run python scripts/eodhd/fetch_eodhd_us_etf_prices.py --universe provider
uv run python scripts/eodhd/fetch_eodhd_us_etf_dividends.py --universe provider
uv run python scripts/eodhd/fetch_eodhd_us_etf_splits.py --universe provider
```

Starter sleeve only:

```powershell
uv run python scripts/eodhd/fetch_eodhd_us_etf_prices.py --universe starter
uv run python scripts/eodhd/fetch_eodhd_us_etf_dividends.py --universe starter
uv run python scripts/eodhd/fetch_eodhd_us_etf_splits.py --universe starter
```

### US index / benchmark reference

```powershell
Set-Location "C:\Users\olegr\PycharmProjects\btest"
uv run python scripts/eodhd/fetch_eodhd_index_ref_universe.py
uv run python scripts/eodhd/fetch_eodhd_index_ref_prices.py
```

### UK/EU ETF

```powershell
Set-Location "C:\Users\olegr\PycharmProjects\btest"
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_universe.py
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_prices.py
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_dividends.py
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_splits.py
```

### UK/EU index / benchmark reference

```powershell
Set-Location "C:\Users\olegr\PycharmProjects\btest"
uv run python scripts/eodhd/fetch_eodhd_uk_eu_index_ref_universe.py
uv run python scripts/eodhd/fetch_eodhd_uk_eu_index_ref_prices.py
```

## Useful targeted reruns

Single ticker:

```powershell
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_prices.py --tickers MAJMEL.CO
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_dividends.py --tickers MAJMEL.CO
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_splits.py --tickers MAJMEL.CO
```

Small smoke batch:

```powershell
uv run python scripts/eodhd/fetch_eodhd_us_etf_prices.py --universe starter --limit 5 --full-refresh
uv run python scripts/eodhd/fetch_eodhd_uk_eu_index_ref_prices.py --limit 5 --full-refresh
```

Windowed rerun:

```powershell
uv run python scripts/eodhd/fetch_eodhd_us_etf_prices.py --from 2026-01-01 --to 2026-05-07
uv run python scripts/eodhd/fetch_eodhd_uk_eu_etf_dividends.py --from 2025-01-01 --to 2026-05-07
```

## Progress / audit checks

Quick cross-lane snapshot:

```powershell
Set-Location "C:\Users\olegr\PycharmProjects\btest"
python -u tmp_poll_eodhd_progress.py
```

For a lane-specific audit, inspect:

- the main parquet output (`prices_daily.parquet`, `dividends_history.parquet`, `splits_history.parquet`),
- the state sidecar (`*_fetch_state.csv`),
- and for event lanes the audit sidecar (`*_fetch_audit.csv`).

Interpretation rules:

- `ok`: provider returned data and the lane recorded rows.
- `empty`: provider returned no dividend/split history for that pair.
- `up_to_date`: the pair was already covered through the requested upper bound and no HTTP request was needed on this rerun.

## Operator notes

- Keep factual counts and completion status in the manifests, not in this runbook.
- Update the relevant manifest whenever a lane changes scope, output paths, or observed on-disk counts materially.
- If a session is interrupted, rerun the same script(s) without `--full-refresh` unless a rebuild is explicitly intended.

