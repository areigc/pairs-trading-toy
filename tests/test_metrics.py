import pandas as pd
import pytest

from pairs_trading.metrics import (
    cumulative_return,
    maximum_drawdown,
    performance_summary,
    trade_count,
)


def test_compounding_and_drawdown() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    assert cumulative_return(returns) == pytest.approx(1.1 * 0.8 * 1.05 - 1.0)
    assert maximum_drawdown(returns) == pytest.approx(-0.20)


def test_trade_count_includes_reversals() -> None:
    positions = pd.Series([0, 1, 1, 0, -1, 1, 0])
    assert trade_count(positions) == 3


def test_flat_strategy_metrics_are_defined() -> None:
    summary = performance_summary(pd.Series([0.0, 0.0]), pd.Series([0, 0]))
    assert summary["cumulative_return"] == 0.0
    assert summary["annualized_volatility"] == 0.0
    assert summary["sharpe_ratio"] == 0.0
    assert summary["maximum_drawdown"] == 0.0
    assert summary["trade_count"] == 0
