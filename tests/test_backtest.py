import numpy as np
import pytest

from pairs_trading.backtest import BacktestConfig, run_backtest
from pairs_trading.data import generate_synthetic_prices, train_test_split


def test_backtest_is_out_of_sample_and_reports_metrics() -> None:
    prices = generate_synthetic_prices(n_periods=500, seed=12)
    training, test = train_test_split(prices)
    result = run_backtest(training, test)

    assert result.frame.index.equals(test.index)
    assert result.frame["gross_return"].iloc[0] == 0.0
    assert np.isfinite(result.frame["equity"]).all()
    assert set(result.metrics) == {
        "cumulative_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "trade_count",
    }
    assert result.metrics["trade_count"] > 0


def test_transaction_costs_are_configurable_and_reduce_period_returns() -> None:
    prices = generate_synthetic_prices(n_periods=500, seed=4)
    training, test = train_test_split(prices)
    free = run_backtest(training, test, config=BacktestConfig(transaction_cost_bps=0.0))
    costly = run_backtest(training, test, config=BacktestConfig(transaction_cost_bps=10.0))

    assert costly.frame["transaction_cost"].sum() > 0
    expected_difference = costly.frame["turnover"] * 10.0 / 10_000.0
    assert np.allclose(
        free.frame["strategy_return"] - costly.frame["strategy_return"],
        expected_difference,
    )


def test_backtest_rejects_overlapping_splits() -> None:
    prices = generate_synthetic_prices(n_periods=100)
    with pytest.raises(ValueError, match="must end before"):
        run_backtest(prices.iloc[:70], prices.iloc[60:])
