import pandas as pd
import pytest

from pairs_trading.data import generate_synthetic_prices, load_price_csv, train_test_split


def test_synthetic_prices_are_reproducible_positive_and_correlated() -> None:
    first = generate_synthetic_prices(n_periods=300, seed=7)
    second = generate_synthetic_prices(n_periods=300, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert (first > 0).all().all()
    assert first.pct_change().corr().iloc[0, 1] > 0.6


def test_train_test_split_is_chronological() -> None:
    prices = generate_synthetic_prices(n_periods=10)
    training, test = train_test_split(prices, train_fraction=0.6)

    assert len(training) == 6
    assert len(test) == 4
    assert training.index.max() < test.index.min()


def test_load_price_csv_renames_columns(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "x": [10.0, 10.5],
            "y": [20.0, 20.4],
        }
    ).to_csv(path, index=False)

    loaded = load_price_csv(path, price_columns=("x", "y"))
    assert list(loaded.columns) == ["asset_a", "asset_b"]
    assert loaded.index.name == "date"


def test_synthetic_input_validation() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_prices(n_periods=2)
