"""Command-line demonstration using synthetic data only."""

from __future__ import annotations

import argparse
from pathlib import Path

from pairs_trading.backtest import BacktestConfig, plot_results, run_backtest
from pairs_trading.data import generate_synthetic_prices, train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the educational pairs-trading backtest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None, help="optional plot path")
    args = parser.parse_args()

    prices = generate_synthetic_prices(seed=args.seed)
    training, test = train_test_split(prices)
    config = BacktestConfig(transaction_cost_bps=args.cost_bps)
    result = run_backtest(training, test, config=config)

    print(f"OLS intercept: {result.model.intercept:.4f}")
    print(f"OLS hedge ratio: {result.model.hedge_ratio:.4f}")
    for name, value in result.metrics.items():
        if name == "trade_count":
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")

    if args.output is not None:
        figure, _ = plot_results(prices, result, entry_z=config.entry_z, exit_z=config.exit_z)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
