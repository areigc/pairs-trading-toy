# Pairs Trading Toy

A small, beginner-friendly Python 3.12+ research project that demonstrates the mechanics of pairs trading with synthetic data. It estimates a hedge ratio on a training period, forms a spread, converts that spread to a rolling z-score, creates trading positions, and evaluates those positions on a separate test period.

> **Educational use only.** This repository is not financial advice, investment research, or a live-trading system. It contains no brokerage integration, credentials, secret handling, live market-data API, or order execution.

## What you will learn

- how to generate two correlated price series with a mean-reverting relationship;
- how ordinary least squares (OLS) estimates a hedge ratio;
- how a spread and rolling z-score turn a relationship into signals;
- how to keep model fitting separate from out-of-sample testing;
- how transaction costs affect simulated performance;
- how to interpret cumulative return, volatility, Sharpe ratio, drawdown, and trade count; and
- why a convincing historical backtest can still fail in the future.

## Project layout

```text
.
├── README.md
├── pyproject.toml
├── examples/
│   └── load_csv.py
├── notebooks/
│   └── pairs_trading_walkthrough.ipynb
├── src/pairs_trading/
│   ├── __init__.py
│   ├── backtest.py
│   ├── data.py
│   ├── demo.py
│   ├── metrics.py
│   └── model.py
└── tests/
    ├── test_backtest.py
    ├── test_data.py
    ├── test_metrics.py
    └── test_model.py
```

## Setup

You need Python 3.12 or newer. The commands below create an isolated environment and install the package plus its learning and test tools.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the automated tests:

```bash
pytest
```

Run the synthetic command-line demo and save a plot:

```bash
pairs-trading-demo --output outputs/backtest.png
```

Open the guided notebook:

```bash
jupyter lab notebooks/pairs_trading_walkthrough.ipynb
```

To execute the notebook non-interactively without modifying the clean source notebook:

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/pairs_trading_walkthrough.ipynb \
  --output /tmp/pairs_trading_walkthrough.executed.ipynb \
  --ExecutePreprocessor.timeout=120
```

## The idea in plain language

Suppose two assets have historically moved together. Rather than predict whether the whole market rises or falls, a pairs strategy watches their relative relationship. If one asset becomes unusually expensive compared with the other, the toy strategy shorts the expensive side and buys the cheap side. It exits when the relationship returns near its recent average.

The default data is deliberately friendly: `asset_a` is a random price path and `asset_b` is constructed from `asset_a` plus a stationary AR(1) residual. Real asset pairs are much messier.

## Mathematics

### 1. Estimate the hedge ratio

On the training observations only, OLS fits

$$
B_t = \alpha + \beta A_t + \varepsilon_t,
$$

where $A_t$ and $B_t$ are the two prices, $\alpha$ is the intercept, $\beta$ is the hedge ratio, and $\varepsilon_t$ is the residual.

The fitted spread is

$$
S_t = B_t - (\hat{\alpha} + \hat{\beta}A_t).
$$

A positive spread says `asset_b` is above the fitted relationship; a negative spread says it is below. OLS association alone does **not** prove that the spread will mean-revert.

### 2. Normalize the spread

Using a trailing window of $w$ observations,

$$
Z_t = \frac{S_t - \bar{S}_{t,w}}{\sigma_{t,w}}.
$$

The rolling mean and standard deviation use information available through time $t$. With the default settings, the strategy enters when $|Z_t| \ge 2.0$ and exits when $|Z_t| \le 0.5$.

### 3. Translate the z-score into positions

- $Z_t \le -2$: long the spread (`+1`), meaning long `asset_b` and short the hedge-adjusted `asset_a` leg.
- $Z_t \ge 2$: short the spread (`-1`).
- While in a trade, hold the state until the z-score moves inside the exit band.
- $|Z_t| \le 0.5$: flat (`0`).

A signal observed at today's close is applied to the **next** period's return. This one-period lag prevents the simulator from earning a return that happened before the signal was known.

### 4. Backtest out of sample

The default chronological split uses 60% of observations for training and 40% for testing. The hedge ratio is estimated once on training data and remains fixed throughout the test. Training history may warm up the trailing z-score, but all reported P&L is from the later test dates.

The long-spread return before costs is approximated by

$$
r^{spread}_t = \frac{r^B_t - \hat{\beta}r^A_t}{1 + |\hat{\beta}|}.
$$

Dividing by gross leg weight makes returns easier to interpret. The backtest subtracts configurable proportional costs whenever the position changes:

$$
cost_t = \frac{bps}{10{,}000}|position_t-position_{t-1}|.
$$

This is still simplified: it omits bid-ask dynamics, slippage variation, borrow fees, financing, margin, partial fills, and market impact.

## Reported metrics

- **Cumulative return:** compounded growth over the test period minus one.
- **Annualized volatility:** daily sample standard deviation multiplied by $\sqrt{252}$.
- **Sharpe ratio:** annualized mean return divided by volatility, assuming a zero risk-free rate.
- **Maximum drawdown:** worst percentage fall from a previous equity peak.
- **Trade count:** entries from flat plus direct position reversals.

The plots show normalized test prices, the OLS spread with its rolling z-score, position state, and the after-cost equity curve.

## Optional historical CSV example

Synthetic data is the default and requires no keys or network access. If you already have a lawful local CSV, keep that workflow separate:

```python
from pairs_trading.data import load_price_csv, train_test_split

prices = load_price_csv(
    "path/to/your/prices.csv",
    date_column="date",
    price_columns=("stock_x", "stock_y"),
)
training_prices, test_prices = train_test_split(prices, train_fraction=0.60)
```

The loader expects one date column and exactly two strictly positive price columns. Do not commit licensed datasets, private data, credentials, or API keys. The repository ignores `data/` by default.

## Biases and failure modes

**Look-ahead bias** occurs when a decision uses data that was unavailable at the simulated decision time. Examples include estimating the hedge ratio with the full dataset, centering today's z-score using future observations, or applying a close-based signal to the same close-to-close return. This project fits only on training data, uses trailing windows, and lags positions by one period. Those safeguards help, but timestamp alignment must be reconsidered for any real dataset.

**Overfitting** means tuning rules so closely to one sample that they capture noise rather than a durable relationship. Searching many z-score windows, thresholds, train/test dates, or cost assumptions and choosing the prettiest result can overfit even when each individual backtest is coded correctly.

**Data snooping** is the repeated reuse of the same observations to invent, reject, and refine hypotheses. The nominal test set stops being truly out of sample once you inspect it repeatedly. A better research process preserves a final untouched holdout or uses carefully designed walk-forward evaluation.

**Survivorship bias** appears when today's surviving securities are used to reconstruct the past while delisted, acquired, bankrupt, or otherwise failed securities are absent. The apparent opportunity set then looks safer and more stable than the one actually available at the time.

**Cointegration can break down.** Even a pair with a historically stationary spread can separate after business-model changes, mergers, capital-structure changes, regulation, index reconstitution, macro shocks, or simple statistical regime change. Correlation is not cointegration, an in-sample cointegration test is not a guarantee, and a fixed hedge ratio can become obsolete.

Other important limitations include selection bias, unrealistic fills, ignored short availability, taxes, latency, parameter instability, non-normal returns, and the favorable construction of the synthetic data.

## Exercises

1. Replace the single train/test split with expanding-window walk-forward re-estimation. Compare it with the fixed model.
2. Sweep transaction costs from 0 to 50 basis points and plot how Sharpe ratio and cumulative return change. Explain why this is a sensitivity study, not permission to choose the best-looking cost.
3. Add an Augmented Dickey-Fuller test for the training spread. Keep the test-period results untouched until the rule is finalized.
4. Add a maximum holding period and record win rate, average trade duration, and per-trade return.
5. Simulate a structural break by changing the synthetic hedge ratio halfway through the test period. Observe how the strategy behaves.
6. Build a point-in-time pair-selection exercise that includes delisted names, then explain how it addresses survivorship bias.

## License

MIT. See [LICENSE](LICENSE).
