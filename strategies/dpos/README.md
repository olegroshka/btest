# Disclosed positioning: the sized runs

Funds disclose positions in public records: the FINRA short-interest and
daily short-volume files and the SEC's 13F holdings in the United States, and
the per-holder net short registers of the European regulators (the FCA until
July 2026, the AMF, the AFM, Finansinspektionen, Finanstilsynet, the Central
Bank of Ireland, the Bundesanzeiger). datacli reads those public sources,
builds point-in-time position panels from them and evaluates whether what
they disclose orders future returns. The strategies in this folder size the
orderings that survived that evaluation, with the engine's costs: commission,
square-root impact, a volume cap and the broker's borrow per name. Every
input is a parquet file written by a datacli script; nothing here reads a
database or a vendor API.

## Data

Each strategy reads its inputs from `data/raw/positioning/exports/btest/`
under the btest root (the UK register check from `exports/btest_uk/`), the
folder datacli writes to when its positioning root is this repo's
`data/raw/positioning`. The folder is a local copy and is not committed:
anyone running these needs the files below in place, or sets
`DATACLI_BTEST_EXPORTS` to the folder that holds them.

| strategy | inputs (the datacli script that writes them) | what it sizes |
|---|---|---|
| `short_accumulation.py` | `prices_daily.parquet`, `short_z_inventory[_raw].parquet` (`scripts/export_btest_inputs.py`) | abnormal short accumulation in the US (the EWM z-score of log short inventory, residualised on reversal and momentum), equal-weight deciles, weekly |
| `short_accumulation_v2.py` | the same prices, `short_z_inventory_size[_neg].parquet`, `borrow_rates.parquet` (`export_btest_inputs.py --second`) | the same ordering, size-residual, as rank-weighted 30 percent books, monthly |
| `long_flow.py` | `prices_daily_long.parquet`, `long_flow_cohort[_neg]`, `long_level_cohort[_neg]`, `long_size_tercile_liquid.parquet` (`--second`) | the hedge-fund cohort's residual quarterly net buying from 13F, long-short books, by size third |
| `register_profit.py` | `register_prices_daily.parquet`, `register_profit[_raw][_neg].parquet`, `register_borrow_rates[_cap50].parquet` (`scripts/export_register_inputs.py [--markets uk] [--rate-cap 0.5]`) | the funds' disclosed short profit on the European registers (FIFO lots per holder from the register's changes, residualised on reversal and momentum), rank-weighted books with borrow per name |
| `register_long_leg.py` | `register_prices_daily.parquet`, `register_long_leg_weights.parquet` (`scripts/register_long_leg.py`) | the long leg of the register ordering alone, weights given (`TargetWeights`) |

Run from the btest root, for example:

```
uv run python strategies/dpos/register_profit.py
DPOS_REBALANCE=1m DPOS_TURNOVER=0.3 DPOS_BORROW_RATES=register_borrow_rates_cap50 uv run python strategies/dpos/register_profit.py
DATACLI_BTEST_EXPORTS=data/raw/positioning/exports/btest_uk DPOS_TAG=uk DPOS_MIN_PRICE=0.5 DPOS_START=2013-01-02 DPOS_END=2026-07-10 DPOS_REBALANCE=1m DPOS_TURNOVER=0.3 DPOS_BORROW_RATES=register_borrow_rates_cap50 uv run python strategies/dpos/register_profit.py
```

Each file's docstring names its environment knobs (`DPOS_*`); outputs go to
`outputs/<strategy name>/` (tearsheet, returns, weights, trades). The
register strategies use the engine's per-name borrow
(`BorrowCost(rates_path=...)`), `Identity` signals,
`LongShortPortfolio(min_names_per_book=...)` and, for the long leg, a
strategy with no factors; those primitives were added for them (see
`AGENT_DSL_REFERENCE.md`).

## Findings (as of 2026-10-04)

- **US short accumulation**: the decile spread of the residual z-score is
  zero before costs and negative after; the second iteration (rank books,
  monthly, size-residual) is zero gross, about -1 percent a year after costs
  at 6.5 times turnover; the broker's rates are immaterial (an
  easy-to-borrow book).
- **US long side**: the cohort's residual net buying orders returns in the
  smallest names, below the 5 million dollar liquidity floor; sized on
  liquid names it is negative before costs.
- **European registers, weekly**: the funds' short profit orders returns on
  every register with enough names, but the pooled euro book loses before
  costs (-1 to -2 percent a year), -4 to -6 after costs and borrow (the
  short book pays about 3 percent a year). Its long leg alone is the
  market: beta one, alpha within two percent of zero.
- **UK register, frozen 2012 to 2026-07**: weekly, 1 to 2 percent a year
  after costs (Sharpe 0.2); monthly with a 0.3 turnover limit, about 5
  percent a year after costs and borrow over 2013 to 2026 at Sharpe 0.6,
  drawdown 16 percent (first half Sharpe 0.9 net, second 0.4). The first
  net-positive sized book of the line, in sample, on a register that no
  longer publishes per holder.
- **Euro registers, the same monthly form from 2013**: positive before 2019
  (Sharpe 0.3 after costs), negative since (-7 percent a year after costs),
  flat before costs over the whole span. The years before 2019 are where the
  mechanism paid on public data.
- **Borrow rates applied backwards**: one snapshot of the broker's rates
  stands for the whole history; a name squeezed today (832 percent a year)
  charged at that rate in years when it was an ordinary short was most of a
  book's cost, so the register runs use the file capped at 50 percent
  (`register_borrow_rates_cap50.parquet`).

The line is parked: the out-of-sample ledgers (the US long side from late
2027, the pooled European registers from April 2027) are read when due, and
the monthly register book above is the one pre-registered sized form to run
next to the European ledger.
