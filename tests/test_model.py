import numpy as np
import pandas as pd
import pytest

from pairs_trading.model import fit_hedge_ratio, generate_positions, rolling_zscore


def test_ols_recovers_known_relationship() -> None:
    asset_a = np.linspace(10.0, 30.0, 100)
    prices = pd.DataFrame({"asset_a": asset_a, "asset_b": 4.0 + 1.5 * asset_a})
    model = fit_hedge_ratio(prices)

    assert model.intercept == pytest.approx(4.0)
    assert model.hedge_ratio == pytest.approx(1.5)
    assert np.abs(model.spread(prices)).max() < 1e-10


def test_rolling_zscore_waits_for_full_window() -> None:
    spread = pd.Series([1.0, 2.0, 3.0, 4.0])
    zscore = rolling_zscore(spread, window=3)

    assert zscore.iloc[:2].isna().all()
    assert zscore.iloc[2] == pytest.approx((3.0 - 2.0) / np.std([1.0, 2.0, 3.0]))


def test_positions_enter_hold_and_exit() -> None:
    zscore = pd.Series([np.nan, -2.1, -1.0, -0.4, 2.2, 0.4])
    positions = generate_positions(zscore, entry_z=2.0, exit_z=0.5)

    assert positions.tolist() == [0, 1, 1, 0, -1, 0]
