"""Optional local-CSV example; this file is not used by the default workflow."""

from pairs_trading.data import load_price_csv, train_test_split

# Expected columns: date, stock_x, stock_y. No API key or network call is used.
prices = load_price_csv(
    "path/to/your/prices.csv",
    date_column="date",
    price_columns=("stock_x", "stock_y"),
)
training_prices, test_prices = train_test_split(prices, train_fraction=0.60)
