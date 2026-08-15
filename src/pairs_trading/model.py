"""Hedge-ratio estimation, spread normalization, and trading signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeModel:
    """OLS relationship ``asset_b = intercept + hedge_ratio * asset_a``."""

    intercept: float
    hedge_ratio: float

    def spread(self, prices: pd.DataFrame) -> pd.Series:
        """Return the residual spread implied by this fixed model."""
        required = {"asset_a", "asset_b"}
        if not required.issubset(prices.columns):
            raise ValueError(f"prices must contain columns {sorted(required)}")
        values = prices["asset_b"] - (
            self.intercept + self.hedge_ratio * prices["asset_a"]
        )
        return values.rename("spread")


def fit_hedge_ratio(training_prices: pd.DataFrame) -> HedgeModel:
    """Estimate intercept and hedge ratio by ordinary least squares."""
    required = {"asset_a", "asset_b"}
    if not required.issubset(training_prices.columns):
        raise ValueError(f"training_prices must contain columns {sorted(required)}")
    if len(training_prices) < 2 or training_prices[["asset_a", "asset_b"]].isna().any().any():
        raise ValueError("training_prices must contain at least two complete rows")

    x = training_prices["asset_a"].to_numpy(dtype=float)
    y = training_prices["asset_b"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("asset_a has insufficient variation for OLS")
    return HedgeModel(intercept=float(coefficients[0]), hedge_ratio=float(coefficients[1]))


def rolling_zscore(spread: pd.Series, *, window: int = 20) -> pd.Series:
    """Standardize a spread using a trailing rolling mean and standard deviation."""
    if window < 2:
        raise ValueError("window must be at least 2")
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (spread - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return zscore.rename("zscore")


def generate_positions(
    zscore: pd.Series, *, entry_z: float = 2.0, exit_z: float = 0.5
) -> pd.Series:
    """Create stateful long-spread (+1), flat (0), and short-spread (-1) positions."""
    if entry_z <= 0:
        raise ValueError("entry_z must be positive")
    if not 0 <= exit_z < entry_z:
        raise ValueError("exit_z must be non-negative and smaller than entry_z")

    state = 0
    positions: list[int] = []
    for value in zscore:
        if pd.isna(value):
            state = 0
        elif state == 0:
            if value <= -entry_z:
                state = 1
            elif value >= entry_z:
                state = -1
        elif abs(value) <= exit_z:
            state = 0
        elif state == 1 and value >= entry_z:
            state = -1
        elif state == -1 and value <= -entry_z:
            state = 1
        positions.append(state)
    return pd.Series(positions, index=zscore.index, name="position", dtype=int)
