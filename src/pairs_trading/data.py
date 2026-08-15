"""Data generation and optional CSV-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("asset_a", "asset_b")


def generate_synthetic_prices(
    n_periods: int = 1_000,
    *,
    seed: int = 42,
    start: str = "2020-01-01",
    intercept: float = 12.0,
    hedge_ratio: float = 0.85,
    common_drift: float = 0.0002,
    common_volatility: float = 0.008,
    spread_persistence: float = 0.92,
    spread_volatility: float = 0.65,
) -> pd.DataFrame:
    """Generate two positive, correlated price series with a stationary spread.

    ``asset_a`` follows a geometric random walk. ``asset_b`` is constructed as
    ``intercept + hedge_ratio * asset_a + spread``, where the spread is an AR(1)
    process. This is deliberately convenient synthetic data, not a claim about
    real markets.
    """
    if n_periods < 3:
        raise ValueError("n_periods must be at least 3")
    if not 0 <= spread_persistence < 1:
        raise ValueError("spread_persistence must be in [0, 1)")
    if hedge_ratio <= 0 or spread_volatility <= 0 or common_volatility <= 0:
        raise ValueError("volatility values and hedge_ratio must be positive")

    rng = np.random.default_rng(seed)
    common_returns = rng.normal(common_drift, common_volatility, n_periods)
    asset_a = 100.0 * np.exp(np.cumsum(common_returns))

    innovations = rng.normal(0.0, spread_volatility, n_periods)
    spread = np.empty(n_periods)
    spread[0] = innovations[0]
    for index in range(1, n_periods):
        spread[index] = spread_persistence * spread[index - 1] + innovations[index]

    asset_b = intercept + hedge_ratio * asset_a + spread
    if np.any(asset_b <= 0):
        raise ValueError("parameters produced non-positive synthetic prices")

    dates = pd.bdate_range(start=start, periods=n_periods, name="date")
    return pd.DataFrame({"asset_a": asset_a, "asset_b": asset_b}, index=dates)


def load_price_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    price_columns: Sequence[str] = PRICE_COLUMNS,
) -> pd.DataFrame:
    """Load a local CSV with one date column and exactly two price columns.

    The function performs only local file I/O. It has no market-data or API-key
    integration. Returned columns are renamed to ``asset_a`` and ``asset_b``.
    """
    if len(price_columns) != 2:
        raise ValueError("price_columns must contain exactly two names")

    frame = pd.read_csv(Path(path), parse_dates=[date_column])
    required = [date_column, *price_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required CSV columns: {missing}")

    prices = frame.loc[:, required].copy()
    prices = prices.rename(
        columns={price_columns[0]: "asset_a", price_columns[1]: "asset_b"}
    )
    prices = prices.set_index(date_column).sort_index()
    prices.index.name = "date"

    if prices.index.has_duplicates:
        raise ValueError("CSV contains duplicate dates")
    for column in PRICE_COLUMNS:
        prices[column] = pd.to_numeric(prices[column], errors="raise")
    if prices.isna().any().any():
        raise ValueError("CSV contains missing prices")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    return prices.astype(float)


def train_test_split(
    prices: pd.DataFrame, *, train_fraction: float = 0.60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prices chronologically; never shuffle time-series observations."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(prices) < 3:
        raise ValueError("prices must contain at least three rows")
    split_index = int(len(prices) * train_fraction)
    if split_index == 0 or split_index == len(prices):
        raise ValueError("train_fraction leaves an empty split")
    return prices.iloc[:split_index].copy(), prices.iloc[split_index:].copy()
