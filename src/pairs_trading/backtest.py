"""Out-of-sample pairs-trading backtest and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from pairs_trading.metrics import performance_summary
from pairs_trading.model import HedgeModel, fit_hedge_ratio, generate_positions, rolling_zscore

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class BacktestConfig:
    """Strategy and simulation settings."""

    zscore_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    transaction_cost_bps: float = 5.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be at least 2")
        if self.entry_z <= 0 or not 0 <= self.exit_z < self.entry_z:
            raise ValueError("require entry_z > exit_z >= 0")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


@dataclass(frozen=True)
class BacktestResult:
    """Fixed training model, test-period time series, and summary metrics."""

    model: HedgeModel
    frame: pd.DataFrame
    metrics: dict[str, float | int]


def run_backtest(
    training_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Fit only on training data and simulate only on later test data.

    Training history is retained solely to warm up the rolling z-score. A signal
    observed at today's close affects the next period's return via a one-period
    position lag, preventing same-bar look-ahead.
    """
    config = config or BacktestConfig()
    if training_prices.empty or test_prices.empty:
        raise ValueError("training_prices and test_prices must both be non-empty")
    if training_prices.index.max() >= test_prices.index.min():
        raise ValueError("training data must end before test data begins")

    model = fit_hedge_ratio(training_prices)
    combined = pd.concat([training_prices, test_prices])
    combined_spread = model.spread(combined)
    combined_zscore = rolling_zscore(combined_spread, window=config.zscore_window)

    frame = test_prices.copy()
    frame["spread"] = combined_spread.reindex(test_prices.index)
    frame["zscore"] = combined_zscore.reindex(test_prices.index)
    frame["position"] = generate_positions(
        frame["zscore"], entry_z=config.entry_z, exit_z=config.exit_z
    )

    returns = frame[["asset_a", "asset_b"]].pct_change().fillna(0.0)
    gross_denominator = 1.0 + abs(model.hedge_ratio)
    spread_return = (
        returns["asset_b"] - model.hedge_ratio * returns["asset_a"]
    ) / gross_denominator
    held_position = frame["position"].shift(1, fill_value=0)
    frame["gross_return"] = held_position * spread_return

    turnover = frame["position"].diff().abs()
    turnover.iloc[0] = abs(frame["position"].iloc[0])
    frame["turnover"] = turnover
    frame["transaction_cost"] = turnover * config.transaction_cost_bps / 10_000.0
    frame["strategy_return"] = frame["gross_return"] - frame["transaction_cost"]
    frame["equity"] = (1.0 + frame["strategy_return"]).cumprod()

    metrics = performance_summary(
        frame["strategy_return"],
        frame["position"],
        periods_per_year=config.periods_per_year,
    )
    return BacktestResult(model=model, frame=frame, metrics=metrics)


def plot_results(
    prices: pd.DataFrame,
    result: BacktestResult,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> tuple["Figure", list["Axes"]]:
    """Plot normalized prices, spread/z-score, positions, and equity curve."""
    import matplotlib.pyplot as plt

    test_prices = prices.reindex(result.frame.index)
    normalized = test_prices / test_prices.iloc[0] * 100.0
    figure, axes_array = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes = list(axes_array)

    normalized.plot(ax=axes[0], title="Test-period prices (normalized to 100)")
    axes[0].set_ylabel("normalized price")

    axes[1].plot(result.frame.index, result.frame["spread"], color="tab:blue", label="spread")
    axes[1].set_ylabel("spread")
    z_axis = axes[1].twinx()
    z_axis.plot(result.frame.index, result.frame["zscore"], color="tab:orange", alpha=0.7, label="z-score")
    for level, style in [(entry_z, "--"), (-entry_z, "--"), (exit_z, ":"), (-exit_z, ":")]:
        z_axis.axhline(level, color="grey", linestyle=style, linewidth=0.8)
    z_axis.set_ylabel("z-score")
    axes[1].set_title("OLS spread and rolling z-score")

    axes[2].step(result.frame.index, result.frame["position"], where="post")
    axes[2].set_yticks([-1, 0, 1], labels=["short", "flat", "long"])
    axes[2].set_title("Spread position")

    axes[3].plot(result.frame.index, result.frame["equity"], color="tab:green")
    axes[3].axhline(1.0, color="grey", linewidth=0.8)
    axes[3].set_ylabel("growth of $1")
    axes[3].set_title("Equity curve after transaction costs")
    axes[3].set_xlabel("date")

    figure.tight_layout()
    return figure, axes
