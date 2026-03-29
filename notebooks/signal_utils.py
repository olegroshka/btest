"""
Core utilities for signal generation - Polars version.

This module provides shared functionality for both macro and micro signals:
- Volatility filter computation
- Signal transformation (apply filter, shift)
- Weight computation
"""
from __future__ import annotations

from datetime import date
from typing import Optional, Tuple, List, Dict, Any

import polars as pl
import numpy as np


# =============================================================================
# Filter Functions
# =============================================================================

def apply_scaling(
    df: pl.DataFrame,
    ewm_mean_hl: int | None = None,
    ewm_std_hl: int | None = None,
    bounds: tuple[float, float] | None = None,
    quantile_bias: float | None = None,
    apply_tanh: bool = False,
) -> pl.DataFrame:
    """Apply z-score normalization and clipping to a signal.
    
    Optionally smooths with ewm_mean, then normalizes by ewm_std
    and clips to bounds.
    
    Args:
        df: DataFrame with columns: date, score
        ewm_mean_hl: Halflife for EWM mean smoothing (None to skip)
        ewm_std_hl: Halflife for EWM std normalization (None to skip)
        bounds: (lower, upper) clip bounds, or None to skip clipping
        quantile_bias: Quantile bias for micro signal (only go short below this value)
        apply_tanh: Whether to apply tanh transformation
        
    Returns:
        DataFrame with columns: date, score
    """
    df = df.select(["date", "score"]).sort("date")
    
    # Step 1: Subtract mean (with optional quantile bias correction)
    if ewm_mean_hl is not None:
        if quantile_bias is not None:
            df = df.with_columns(
                score=pl.col("score")
                - pl.col("score").rolling_quantile(
                    quantile_bias, window_size=ewm_mean_hl
                )
            )
        else:
            df = df.with_columns(
                score=pl.col("score") - pl.col("score").ewm_mean(half_life=ewm_mean_hl)
            )
    
    # Step 2: Normalize by EWM std
    if ewm_std_hl is not None:
        df = df.with_columns(
            score=pl.col("score") / pl.col("score").ewm_std(half_life=ewm_std_hl)
        )
    
    # Step 3: Clip to bounds
    if bounds is not None:
        df = df.with_columns(score=pl.col("score").clip(bounds[0], bounds[1]))
    
    # Step 4: Apply tanh squashing
    if apply_tanh:
        df = df.with_columns(score=pl.col("score").tanh())
    
    return df


def apply_filter(
    score_df: pl.DataFrame,
    filter_df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply volatility filter to a score.
    
    Joins score with filter DataFrame and multiplies score by filter value.
    Filter is forward-filled to handle missing dates.
    
    Args:
        score_df: DataFrame with columns: date, score
        filter_df: DataFrame with columns: date, filter
        
    Returns:
        DataFrame with columns: date, score
    """
    df = score_df.join(filter_df, on="date", how="left")
    df = df.with_columns(pl.col("filter").fill_null(strategy="forward"))
    df = df.with_columns((pl.col("score") * pl.col("filter")).alias("score"))
    return df.select(["date", "score"])


def compute_inv_vol_weights(
    start_date: date,
    end_date: date,
    jade_config: Dict[str, Any],
    returns_config: Dict[str, Any],
    hl: int = 30,
) -> pl.DataFrame:
    """Compute inverse-vol weights per asset pair.
    
    Loads index weights, computes composite return per asset pair,
    then applies EWM inverse-vol weighting (1 / ewm_std).
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        jade_config: Config dict with jade_tables section
        returns_config: Config dict with returns_streams section
        hl: Halflife for EWM std calculation (default 30)
        
    Returns:
        DataFrame with columns: date, asset_pair, inv_vol_w
    """
    # In production: index_df = compute_indices(start_date, end_date, ...)
    # index_df has columns: date, asset_pair, asset, hedge_w, total_return
    
    # Composite return per asset pair: sum(hedge_w * total_return)
    # pair_returns = (
    #     index_df.sort("date", "asset_pair", "asset")
    #     .group_by("date", "asset_pair", maintain_order=True)
    #     .agg(total_return=(pl.col("hedge_w") * pl.col("total_return")).sum())
    #     .drop_nulls("total_return")
    # )
    
    # Inverse-vol weight per asset pair
    # df = pair_returns.sort("date", "asset_pair")
    # df = df.with_columns(pl.col("total_return").fill_nan(None))
    # df = df.with_columns(
    #     inv_vol_scale=1.0
    #     / pl.col("total_return").ewm_std(half_life=hl).over("asset_pair")
    # )
    # df = df.with_columns(
    #     pl.col("inv_vol_scale")
    #     .replace((float("inf"), float("-inf")), None)
    #     .fill_nan(None)
    # )
    # return df.select("date", "asset_pair", "inv_vol_scale")
    
    # Placeholder for demonstration
    dates = pl.date_range(start_date, end_date, eager=True)
    return pl.DataFrame({
        "date": dates,
        "inv_vol_scale": [1.0] * len(dates),
    })


def apply_signal_to_asset(
    score_df: pl.DataFrame,
    config: dict,
) -> pl.DataFrame:
    """Expand a signal to all asset pairs with index weights and inverse-vol weighting.
    
    Args:
        score_df: DataFrame with columns [date, score] (universal) or
                  [date, asset_pair, score] (per-pair routing).
        config: Full YAML config dict
        
    Returns:
        DataFrame with columns: date, asset_pair, asset, hedge_w, total_return, score, inv_vol_scale
    """
    start_dt = date.fromisoformat(config["start_date"])
    end_dt = date.fromisoformat(config["end_date"])
    lag = config["lag"]
    config_data = config["data"]
    
    # Get inverse-vol weights
    ap_weight_df = compute_inv_vol_weights(
        start_dt,
        end_dt,
        jade_config=config_data["jade_tables"],
        returns_config=config_data["returns_streams"],
        hl=config["hl_asset_vol"],
    )
    
    # Join with score
    join_on = ["date", "asset_pair"] if "asset_pair" in score_df.columns else ["date"]
    signal_df = score_df.join(ap_weight_df, on=join_on, how="left")
    signal_df = signal_df.with_columns(pl.col("inv_vol_scale").fill_nan(None))
    
    # Compute weighted score: w = hedge_w * score * inv_vol_scale
    signal_df = signal_df.with_columns(
        w=(pl.col("hedge_w") * pl.col("score") * pl.col("inv_vol_scale"))
    )
    
    # Vol-scale w to target annualized volatility (default 10%)
    target_vol = config.get("target_vol", 0.10)
    signal_return = (
        signal_df.sort("date", "asset_pair", "asset")
        .with_columns(pl.col("w").shift(lag).over("asset_pair", "asset"))
        .group_by("date", maintain_order=True)
        .agg(signal_return=(pl.col("w") * pl.col("total_return")).sum())
    )
    signal_return = signal_return.with_columns(
        pl.when(pl.col("signal_return") != 0)
        .then(pl.col("signal_return"))
        .otherwise(None)
        .alias("signal_return")
    )
    signal_return = signal_return.with_columns(
        signal_std=pl.col("signal_return").rolling_std(
            window_size=signal_return.height, min_samples=252
        )
    )
    signal_return = signal_return.with_columns(pl.col("signal_std").fill_nan(None))
    signal_return = signal_return.with_columns(
        vol_scale=pl.lit(target_vol) / (pl.col("signal_std") * (252**0.5))
    )
    signal_return = signal_return.with_columns(
        pl.col("vol_scale").replace((float("inf"), float("-inf")), None).fill_nan(None)
    )
    
    # Apply vol scaling
    signal_df = signal_df.join(
        signal_return.select("date", "vol_scale"), on="date", how="left"
    )
    signal_df = signal_df.with_columns(w=pl.col("w") * pl.col("vol_scale"))
    signal_df = signal_df.drop("vol_scale")
    
    return signal_df.sort("date", "asset_pair", "asset")


def get_filter(
    start_date: date,
    end_date: date,
    config_data: dict,
    level: float = 0.75,
    window: int = 45,
    asset: str = "es1",
) -> pl.DataFrame:
    """Create ES-based volatility filter.
    
    Filters out high-volatility days where intraday range exceeds rolling quantile.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        config_data: Config dict with jade_tables section
        level: Quantile threshold (default 0.75)
        window: Rolling window size (default 45)
        asset: Asset to filter on (default 'es1')
        
    Returns:
        DataFrame with columns: date, filter (1.0 for normal days, 0.0 for high-vol)
    """
    # In production: load futures data
    # fut_config = config_data["jade_tables"]["bbg_futures"]
    # fut_df = read_futures_data(
    #     start_date=start_date,
    #     end_date=end_date,
    #     table=fut_config["table"],
    #     tickers=fut_config["tickers"],
    # )
    # df = fut_df.filter(pl.col("asset") == asset).clone().sort("date")
    
    # Placeholder for demonstration - in production uses actual OHLC data
    dates = pl.date_range(start_date, end_date, eager=True)
    df = pl.DataFrame({
        "date": dates,
        "open": [100.0] * len(dates),
        "high": [101.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [100.5] * len(dates),
    })
    
    # Compute normalized intraday range
    df = df.with_columns(
        ((pl.col("high") - pl.col("low")) / (0.5 * (pl.col("open") + pl.col("close"))))
        .fill_null(0.0)
        .alias("m")
    )
    
    # Compute rolling quantile
    df = df.with_columns(
        pl.col("m").rolling_quantile(level, window_size=window).alias("m_quantile")
    )
    
    # Set filter: 0 when range exceeds quantile, else sign(m)
    df = df.with_columns(
        pl.when(pl.col("m") > pl.col("m_quantile"))
        .then(0.0)
        .otherwise(pl.col("m").sign())
        .alias("filter")
    )
    
    return df.select(["date", "filter"])


def compute_signal_returns(
    signal_df: pl.DataFrame,
    lag: int = 2,
) -> pl.DataFrame:
    """Compute daily returns using the vol-scaled w column: lag(w) * total_return.
    
    Args:
        signal_df: DataFrame with columns: date, signal_name, asset_pair, asset, w, total_return
        lag: Number of days to lag the weight (default 2)
        
    Returns:
        DataFrame with columns: date, signal_name, signal_return
    """
    df = signal_df.sort("date", "signal_name", "asset_pair", "asset")
    df = df.with_columns(
        pl.col("w").shift(lag).over("signal_name", "asset_pair", "asset"),
        pl.col("total_return")
        .replace((float("inf"), float("-inf")), None)
        .fill_nan(None),
    )
    df = df.with_columns(signal_return=pl.col("w") * pl.col("total_return"))
    df = df.drop_nulls("signal_return")
    df = df.group_by("date", "signal_name").agg(pl.col("signal_return").sum())
    df = df.sort("date", "signal_name")
    
    return df


def zscore_ewm(
    df: pl.DataFrame,
    value_col: str,
    hl_mean: int,
    hl_std: int,
    output_col: str = "score",
) -> pl.DataFrame:
    """Compute EWM z-score normalization.
    
    Pattern: (value - ewm_mean) / ewm_std
    
    Args:
        df: Input DataFrame
        value_col: Column to normalize
        hl_mean: Half-life for mean calculation
        hl_std: Half-life for std calculation
        output_col: Output column name
        
    Returns:
        DataFrame with z-scored values
    """
    return df.with_columns(
        (
            (pl.col(value_col) - pl.col(value_col).ewm_mean(half_life=hl_mean))
            / pl.col(value_col).ewm_std(half_life=hl_std)
        ).alias(output_col)
    )


def compute_rolling_zscore(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    output_col: str = "score",
) -> pl.DataFrame:
    """Compute rolling z-score normalization.
    
    Args:
        df: Input DataFrame
        value_col: Column to normalize
        window: Rolling window size
        output_col: Output column name
        
    Returns:
        DataFrame with z-scored values
    """
    return df.with_columns(
        (
            (pl.col(value_col) - pl.col(value_col).rolling_mean(window))
            / pl.col(value_col).rolling_std(window)
        ).alias(output_col)
    )


# =============================================================================
# KALMAN FILTER UTILITIES
# =============================================================================

def compute_kalman_beta(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 0.01,
    use_constant: bool = False,
) -> np.ndarray:
    """Vectorized Kalman filter using einsum. Handles single or multiple entities.
    
    Model: y_t = beta_t * x_t + (alpha_t) + noise
    where beta_t follows a random walk.
    
    Args:
        x: (n_obs,) or (n_obs, n_entities) independent variable
        y: (n_obs,) or (n_obs, n_entities) dependent variable
        delta: Controls transition covariance Q
        use_constant: If True, estimate intercept alongside beta
        
    Returns:
        state_means: (n_obs, n_entities, n_dim) array
            n_dim=1 if use_constant=False (slope only), n_dim=2 if True (slope, intercept)
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
    
    n_obs, n_entities = x.shape
    n_dim = 2 if use_constant else 1
    
    state_means = np.zeros((n_obs, n_entities, n_dim))
    state_covs = np.tile(np.ones((n_dim, n_dim))[np.newaxis, :, :], (n_entities, 1, 1))
    
    F = np.eye(n_dim)
    Q = delta * np.eye(n_dim)
    R = 1.0
    
    for t in range(n_obs):
        if use_constant:
            H = np.stack([x[t], np.ones(n_entities)], axis=1)
        else:
            H = x[t][:, np.newaxis]
        
        if t > 0:
            state_means[t] = np.einsum("ej,jk->ek", state_means[t - 1], F)
            state_covs = np.einsum("ij,ejk,kl->eil", F, state_covs, F) + Q
        
        y_pred = np.einsum("ej,ej->e", H, state_means[t])
        innovation = y[t] - y_pred
        S = np.einsum("ei,eij,ej->e", H, state_covs, H) + R
        K = np.einsum("eij,ej,e->ei", state_covs, H, 1 / S)
        state_means[t] += np.einsum("ei,e->ei", K, innovation)
        I_KH = np.eye(n_dim)[np.newaxis, :, :] - np.einsum("ei,ej->eij", K, H)
        state_covs = np.einsum("eij,ejk->eik", I_KH, state_covs)
    
    return state_means


class KalmanFilter1D:
    """1D Kalman filter for signal smoothing.
    
    Simple implementation for time series smoothing with
    observation and process noise.
    """
    
    def __init__(
        self,
        process_variance: float = 1e-5,
        observation_variance: float = 1e-2,
        initial_estimate: float = 0.0,
        initial_error: float = 1.0,
    ):
        """Initialize Kalman filter.
        
        Args:
            process_variance: Q - process noise variance
            observation_variance: R - observation noise variance
            initial_estimate: Initial state estimate
            initial_error: Initial error covariance
        """
        self.Q = process_variance
        self.R = observation_variance
        self.x = initial_estimate  # state estimate
        self.P = initial_error     # error covariance
        
    def update(self, measurement: float) -> float:
        """Update filter with new measurement.
        
        Args:
            measurement: New observation
            
        Returns:
            Updated state estimate
        """
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x
    
    def filter_series(self, values: np.ndarray) -> np.ndarray:
        """Filter entire series.
        
        Args:
            values: Array of observations
            
        Returns:
            Array of filtered values
        """
        filtered = np.zeros_like(values)
        for i, v in enumerate(values):
            if np.isnan(v):
                filtered[i] = self.x
            else:
                filtered[i] = self.update(v)
        return filtered


def apply_kalman_filter(
    df: pl.DataFrame,
    value_col: str,
    process_var: float = 1e-5,
    obs_var: float = 1e-2,
    output_col: str = "kalman_filtered",
) -> pl.DataFrame:
    """Apply Kalman filter to DataFrame column.
    
    Args:
        df: Input DataFrame
        value_col: Column to filter
        process_var: Process noise variance
        obs_var: Observation noise variance
        output_col: Output column name
        
    Returns:
        DataFrame with filtered column
    """
    values = df[value_col].to_numpy()
    kf = KalmanFilter1D(process_variance=process_var, observation_variance=obs_var)
    filtered = kf.filter_series(values)
    
    return df.with_columns(
        pl.Series(name=output_col, values=filtered)
    )


def kalman_smooth_spread(
    df: pl.DataFrame,
    spread_col: str = "spread",
    process_var: float = 1e-5,
    obs_var: float = 1e-2,
) -> pl.DataFrame:
    """Kalman smooth credit spread for signal computation.
    
    Specialized function for spread smoothing with typical
    credit spread parameters.
    
    Args:
        df: DataFrame with spread data
        spread_col: Name of spread column
        process_var: Process variance (default tuned for spreads)
        obs_var: Observation variance
        
    Returns:
        DataFrame with kalman_spread column
    """
    return apply_kalman_filter(
        df, spread_col, process_var, obs_var, "kalman_spread"
    )


# =============================================================================
# ADDITIONAL SIGNAL HELPERS
# =============================================================================

def compute_momentum(
    df: pl.DataFrame,
    price_col: str,
    lookback: int,
    output_col: str = "momentum",
) -> pl.DataFrame:
    """Compute price momentum (return over lookback period).
    
    Args:
        df: DataFrame with price data
        price_col: Price column name
        lookback: Lookback period in days
        output_col: Output column name
        
    Returns:
        DataFrame with momentum column
    """
    return df.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(lookback) - 1)
        .alias(output_col)
    )


def compute_volatility(
    df: pl.DataFrame,
    return_col: str,
    window: int,
    output_col: str = "volatility",
    annualize: bool = True,
) -> pl.DataFrame:
    """Compute rolling volatility.
    
    Args:
        df: DataFrame with return data
        return_col: Return column name
        window: Rolling window size
        output_col: Output column name
        annualize: Whether to annualize (sqrt(252))
        
    Returns:
        DataFrame with volatility column
    """
    factor = np.sqrt(252) if annualize else 1.0
    return df.with_columns(
        (pl.col(return_col).rolling_std(window) * factor).alias(output_col)
    )


def compute_correlation(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    window: int,
    output_col: str = "correlation",
) -> pl.DataFrame:
    """Compute rolling correlation between two columns.
    
    Args:
        df: Input DataFrame
        col1: First column name
        col2: Second column name  
        window: Rolling window size
        output_col: Output column name
        
    Returns:
        DataFrame with correlation column
    """
    return df.with_columns(
        pl.rolling_corr(col1, col2, window_size=window).alias(output_col)
    )


def sign_flip_for_pair(
    df: pl.DataFrame,
    pair_name: str,
    flip_patterns: List[str] = ["bkln|rty1", "lqd_rh|esi"],
    score_col: str = "score",
) -> pl.DataFrame:
    """Apply sign flip for specific asset pairs.
    
    Some pairs have inverted signal semantics and need
    their scores negated.
    
    Args:
        df: DataFrame with score column
        pair_name: Asset pair name to check
        flip_patterns: List of pattern strings that should be flipped
        score_col: Score column name
        
    Returns:
        DataFrame with potentially flipped scores
    """
    import re
    
    should_flip = any(
        re.search(pattern, pair_name, re.IGNORECASE)
        for pattern in flip_patterns
    )
    
    if should_flip:
        return df.with_columns(
            (-pl.col(score_col)).alias(score_col)
        )
    return df


def route_to_index(
    pair_name: str,
    main_patterns: List[str] = ["main|vg1", "xover|vg1"],
) -> str:
    """Determine data source index for asset pair.
    
    Routes pairs to either 'main' or 'cdxig' data source
    based on pattern matching.
    
    Args:
        pair_name: Asset pair name
        main_patterns: Patterns that route to 'main' source
        
    Returns:
        'main' or 'cdxig'
    """
    if pair_name in main_patterns:
        return "main"
    return "cdxig"
