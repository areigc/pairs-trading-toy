"""Educational pairs-trading research helpers."""

from pairs_trading.backtest import BacktestConfig, BacktestResult, run_backtest
from pairs_trading.data import generate_synthetic_prices, load_price_csv, train_test_split
from pairs_trading.model import HedgeModel, fit_hedge_ratio, generate_positions, rolling_zscore

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "HedgeModel",
    "fit_hedge_ratio",
    "generate_positions",
    "generate_synthetic_prices",
    "load_price_csv",
    "rolling_zscore",
    "run_backtest",
    "train_test_split",
]
