"""Performance metrics for the toy backtest."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cumulative_return(returns: pd.Series) -> float:
    """Compound simple period returns."""
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def annualized_volatility(returns: pd.Series, *, periods_per_year: int = 252) -> float:
    """Annualize sample return volatility."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, *, periods_per_year: int = 252) -> float:
    """Annualized zero-risk-free-rate Sharpe ratio."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    volatility = clean.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return float(clean.mean() / volatility * math.sqrt(periods_per_year))


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the worst peak-to-trough equity decline as a non-positive number."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def trade_count(positions: pd.Series) -> int:
    """Count entries from flat plus direct reversals as new trades."""
    current = positions.fillna(0).astype(int)
    previous = current.shift(1, fill_value=0)
    entries = (current != 0) & ((previous == 0) | (current != previous))
    return int(entries.sum())


def performance_summary(
    returns: pd.Series, positions: pd.Series, *, periods_per_year: int = 252
) -> dict[str, float | int]:
    """Compute the headline metrics requested by the walkthrough."""
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_volatility": annualized_volatility(
            returns, periods_per_year=periods_per_year
        ),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "maximum_drawdown": maximum_drawdown(returns),
        "trade_count": trade_count(positions),
    }
