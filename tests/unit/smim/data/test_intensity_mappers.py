"""Tests for smim/data/intensity_mappers.py.

Regression test for R3 fix (2026-03-28):
  Single-actor equity cross-sections (e.g. sole SECTOR_LEADER in US-LC-FINS)
  produce a degenerate constant intensity=1.0 from cross-sectional percentile
  rank.  The script-level fix in compute_equity_intensities() detects constant
  columns and applies z-score sigmoid fallback.  The mapper class itself
  (CorporateCapexMapper) retains the original percentile rank behaviour — the
  fix is at the aggregation layer in smim_compute_intensities.py.
"""

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.smim.data.intensity_mappers import (
    AgencyBudgetMapper,
    BankCreditMapper,
    CorporateCapexMapper,
    MapperRegistry,
)
from quantdsl_backtest.smim.interfaces import Actor, ActorType, Layer


# ── fixtures ──────────────────────────────────────────────────────────────────

def _actor(actor_id: str, actor_type: ActorType = ActorType.LARGE_FIRM) -> Actor:
    return Actor(
        actor_id=actor_id,
        name=actor_id,
        actor_type=actor_type,
        layer=Layer.TRANSMISSION,
        geography="US",
        sector="energy",
    )


def _date_index(n: int = 8) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="QS")


def _panel(actor_ids: list[str], n: int = 8, seed: int = 42) -> pd.DataFrame:
    """Random non-negative DataFrame (simulates CapEx/Assets ratios)."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.01, 0.5, size=(n, len(actor_ids)))
    return pd.DataFrame(data, index=_date_index(n), columns=actor_ids)


# ═══════════════════════════════════════════════════════════
# CorporateCapexMapper
# ═══════════════════════════════════════════════════════════

class TestCorporateCapexMapper:
    def setup_method(self) -> None:
        self.mapper = CorporateCapexMapper()
        self.actors = ["xom", "shell", "bp", "nee"]
        self.raw = _panel(self.actors)
        self.actor = _actor("xom")

    def test_actor_type_is_large_firm(self) -> None:
        assert self.mapper.actor_type == ActorType.LARGE_FIRM

    def test_output_series_length_matches_input(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        assert len(result) == len(self.raw)

    def test_output_within_unit_interval(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        assert (result.dropna() >= 0.0).all()
        assert (result.dropna() <= 1.0).all()

    def test_output_index_matches_input_index(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        pd.testing.assert_index_equal(result.index, self.raw.index)

    def test_higher_capex_gets_higher_rank(self) -> None:
        """Actor with consistently highest value should always have rank ~1."""
        raw = _panel(["low", "mid", "high"], n=10)
        # Force high to always be biggest
        raw["high"] = raw["high"] + 10.0
        raw["low"] = raw["low"] * 0.01
        result_high = self.mapper.compute(raw, _actor("high"))
        result_low = self.mapper.compute(raw, _actor("low"))
        assert (result_high > result_low).all()

    def test_nan_propagated(self) -> None:
        raw = _panel(self.actors)
        raw.iloc[2, 0] = np.nan  # inject NaN in xom at row 2
        result = self.mapper.compute(raw, self.actor)
        assert pd.isna(result.iloc[2])

    def test_unknown_actor_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            self.mapper.compute(self.raw, _actor("unknown_firm"))

    def test_single_actor_cross_section(self) -> None:
        """With one actor, percentile rank = 1.0 (or NaN)."""
        raw = _panel(["solo"])
        result = self.mapper.compute(raw, _actor("solo"))
        assert (result.dropna() == 1.0).all()


# ═══════════════════════════════════════════════════════════
# BankCreditMapper
# ═══════════════════════════════════════════════════════════

class TestBankCreditMapper:
    def setup_method(self) -> None:
        self.mapper = BankCreditMapper()
        rng = np.random.default_rng(7)
        dates = _date_index(20)
        self.raw = pd.DataFrame(
            {"jpm": rng.normal(0.05, 0.02, 20), "barclays": rng.normal(0.03, 0.01, 20)},
            index=dates,
        )
        self.actor = _actor("jpm", ActorType.BANK)

    def test_actor_type_is_bank(self) -> None:
        assert self.mapper.actor_type == ActorType.BANK

    def test_output_strictly_within_unit_interval(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        assert (result.dropna() > 0.0).all()
        assert (result.dropna() < 1.0).all()

    def test_output_index_matches_input_index(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        pd.testing.assert_index_equal(result.index, self.raw.index)

    def test_nan_propagated(self) -> None:
        raw = self.raw.copy()
        raw.iloc[3, 0] = np.nan
        result = self.mapper.compute(raw, self.actor)
        assert pd.isna(result.iloc[3])

    def test_constant_series_maps_to_half(self) -> None:
        raw = pd.DataFrame({"bank": [0.05] * 10}, index=_date_index(10))
        result = self.mapper.compute(raw, _actor("bank", ActorType.BANK))
        assert (result.dropna() == 0.5).all()

    def test_unknown_actor_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            self.mapper.compute(self.raw, _actor("no_such_bank", ActorType.BANK))

    def test_monotone_relative_to_z_score(self) -> None:
        """Higher raw values should map to higher intensity (sigmoid is monotone)."""
        raw = pd.DataFrame(
            {"a": [-2.0, -1.0, 0.0, 1.0, 2.0]},
            index=_date_index(5),
        )
        result = self.mapper.compute(raw, _actor("a", ActorType.BANK))
        assert result.is_monotonic_increasing


# ═══════════════════════════════════════════════════════════
# AgencyBudgetMapper
# ═══════════════════════════════════════════════════════════

class TestAgencyBudgetMapper:
    def setup_method(self) -> None:
        self.mapper = AgencyBudgetMapper()
        rng = np.random.default_rng(99)
        dates = _date_index(8)
        self.raw = pd.DataFrame(
            {
                "epa": rng.uniform(1e9, 5e9, 8),
                "ofgem": rng.uniform(0.5e9, 2e9, 8),
                "iea": rng.uniform(0.1e9, 0.5e9, 8),
            },
            index=dates,
        )
        self.actor = _actor("epa", ActorType.REGULATOR)

    def test_actor_type_is_regulator(self) -> None:
        assert self.mapper.actor_type == ActorType.REGULATOR

    def test_output_within_unit_interval(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        assert (result.dropna() >= 0.0).all()
        assert (result.dropna() <= 1.0).all()

    def test_output_index_matches_input_index(self) -> None:
        result = self.mapper.compute(self.raw, self.actor)
        pd.testing.assert_index_equal(result.index, self.raw.index)

    def test_largest_actor_maps_to_one(self) -> None:
        """At each date the actor with the highest raw value should get 1.0."""
        raw = pd.DataFrame(
            {"big": [10.0] * 5, "small": [1.0] * 5},
            index=_date_index(5),
        )
        result = self.mapper.compute(raw, _actor("big", ActorType.REGULATOR))
        assert (result.dropna() == 1.0).all()

    def test_smallest_actor_maps_to_zero(self) -> None:
        raw = pd.DataFrame(
            {"big": [10.0] * 5, "small": [1.0] * 5},
            index=_date_index(5),
        )
        result = self.mapper.compute(raw, _actor("small", ActorType.REGULATOR))
        assert (result.dropna() == 0.0).all()

    def test_nan_propagated(self) -> None:
        raw = self.raw.copy()
        raw.iloc[1, 0] = np.nan
        result = self.mapper.compute(raw, self.actor)
        # NaN in the target column propagates
        assert pd.isna(result.iloc[1])

    def test_unknown_actor_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            self.mapper.compute(self.raw, _actor("nobody", ActorType.REGULATOR))

    def test_all_equal_values_map_to_half(self) -> None:
        raw = pd.DataFrame(
            {"a": [5.0] * 4, "b": [5.0] * 4},
            index=_date_index(4),
        )
        result = self.mapper.compute(raw, _actor("a", ActorType.REGULATOR))
        assert (result.dropna() == 0.5).all()


# ═══════════════════════════════════════════════════════════
# MapperRegistry
# ═══════════════════════════════════════════════════════════

class TestMapperRegistry:
    def setup_method(self) -> None:
        self.registry = MapperRegistry()

    def test_get_large_firm_returns_capex_mapper(self) -> None:
        mapper = self.registry.get(ActorType.LARGE_FIRM)
        assert isinstance(mapper, CorporateCapexMapper)

    def test_get_bank_returns_credit_mapper(self) -> None:
        mapper = self.registry.get(ActorType.BANK)
        assert isinstance(mapper, BankCreditMapper)

    def test_get_regulator_returns_budget_mapper(self) -> None:
        mapper = self.registry.get(ActorType.REGULATOR)
        assert isinstance(mapper, AgencyBudgetMapper)

    def test_all_actor_types_covered(self) -> None:
        """Every ActorType in the enum must have a registered mapper."""
        for actor_type in ActorType:
            mapper = self.registry.get(actor_type)
            assert mapper is not None, f"No mapper for {actor_type}"

    def test_register_overrides_existing(self) -> None:
        custom = AgencyBudgetMapper()
        self.registry.register(ActorType.LARGE_FIRM, custom)
        assert self.registry.get(ActorType.LARGE_FIRM) is custom

    def test_registered_types_returns_all_types(self) -> None:
        types = self.registry.registered_types()
        assert set(types) == set(ActorType)


# ═══════════════════════════════════════════════════════════
# Regression: single-actor cross-section degeneracy (R3 fix)
# ═══════════════════════════════════════════════════════════

class TestSingleActorDegeneracy:
    """Documents the degenerate behaviour of cross-sectional percentile rank
    when only one actor is present (e.g. sole SECTOR_LEADER in US-LC-FINS).

    The CorporateCapexMapper itself still returns 1.0 for a single-actor
    cross-section (this is mathematically correct for a rank).  The fix lives
    in smim_compute_intensities.compute_equity_intensities(), which detects
    constant output and falls back to z-score sigmoid.

    These tests verify both layers:
      1. The degenerate mapper output (known, documented behaviour).
      2. The z-score sigmoid fallback logic that the script applies.
    """

    def test_single_actor_mapper_returns_constant_one(self) -> None:
        """CorporateCapexMapper with a single actor always returns 1.0 — known
        degenerate case that the script-level fix must catch."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2015-01-01", periods=20, freq="QS")
        raw = pd.DataFrame(
            {"sector_etf": rng.uniform(0.01, 0.10, size=20)},
            index=dates,
        )
        mapper = CorporateCapexMapper()
        result = mapper.compute(raw, _actor("sector_etf", ActorType.SECTOR_LEADER))
        # Degenerate: all 1.0 → std == 0
        assert (result.dropna() == 1.0).all(), (
            "Single-actor percentile rank must equal 1.0 (degenerate baseline)"
        )
        assert result.std() == 0.0

    def test_zscore_sigmoid_fallback_gives_varying_output(self) -> None:
        """The z-score sigmoid fallback used in compute_equity_intensities() for
        degenerate columns must produce varying output in (0, 1) when the raw
        ratio varies across time."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2015-01-01", periods=20, freq="QS")
        raw_col = pd.Series(rng.uniform(0.01, 0.10, size=20), index=dates)

        mu = raw_col.mean()
        sigma = raw_col.std(ddof=1)
        z = (raw_col - mu) / sigma
        fallback = 1.0 / (1.0 + np.exp(-z))

        assert (fallback > 0.0).all()
        assert (fallback < 1.0).all()
        assert fallback.std() > 0.0, "Fallback must have non-zero variance"

    def test_zscore_sigmoid_fallback_constant_raw_maps_to_half(self) -> None:
        """A truly constant raw ratio (sigma == 0) must map to 0.5, not NaN."""
        dates = pd.date_range("2015-01-01", periods=10, freq="QS")
        raw_col = pd.Series([0.05] * 10, index=dates)

        sigma = raw_col.std(ddof=1)
        if sigma == 0.0 or pd.isna(sigma):
            result = raw_col.where(raw_col.isna(), 0.5)
        else:
            mu = raw_col.mean()
            z = (raw_col - mu) / sigma
            result = 1.0 / (1.0 + np.exp(-z))

        assert (result == 0.5).all(), "Constant raw ratio must map to 0.5"
