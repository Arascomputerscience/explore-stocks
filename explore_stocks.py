import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# --------------------------------------------------
# SETTINGS (FILL IN YOUR ALPACA KEYS HERE)
# --------------------------------------------------

DB_PATH = Path(r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite")
DB_URI = "file:////192.168.1.95/CS-CLASSROOM/intraday_5m.sqlite?mode=ro&immutable=1"

ALPACA_API_KEY = "PK776L5EFDVIGWY2S5X2N2DZUM"
ALPACA_SECRET_KEY = "rydUMA2mD4DPaA13nsQGiWtjW9WqtN1kiRRkNdiJqqv"

# Replace "SPY" with your exact S&P 500 ticker from Alpaca
SYMBOLS_TO_TRADE = ["CMG", "SPY", "TTWO"]
BUY_THRESHOLD = 0.05       # Buy 1 share if predicted 15m return > 0.05%
LOOP_SECONDS = 300         # 300 seconds = 5 minutes
AUTO_SELL_AFTER_15M = True # Automatically close bot-opened trades after 15 minutes


# --------------------------------------------------
# HELPER: ENGINEER NORMALIZED FEATURES
# --------------------------------------------------

def add_normalized_features(df, groups):
    """Creates scale-free percentage & ratio features so the tree works across any price/split."""
    df["ret_5m"] = groups["close"].pct_change(1) * 100
    df["ret_15m"] = groups["close"].pct_change(3) * 100
    df["range_pct"] = ((df["high"] - df["low"]) / df["open"]) * 100

    rolling_vol = groups["volume"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    df["rel_volume"] = df["volume"] / rolling_vol.replace(0, np.nan)

    return ["ret_5m", "ret_15m", "range_pct", "rel_volume"]


# --------------------------------------------------
# 1. TRAIN DECISION TREE ONCE AT STARTUP
# --------------------------------------------------

def build_and_train_tree():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot see {DB_PATH}. Run: net use \\\\192.168.1.95\\CS-CLASSROOM /user:CS-Students"
        )

    sql = """
        SELECT symbol, ts_utc, open, high, low, close, volume
        FROM bars_5m
        WHERE symbol IN ('AAPL', 'MSFT', 'NVDA')
          AND ts_utc >= '2024-01-01'
          AND ts_utc < '2025-07-01'
        ORDER BY symbol, ts_utc
    """

    with sqlite3.connect(DB_URI, uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")
        df = pd.read_sql_query(sql, conn)

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["trading_date"] = df["ts_utc"].dt.tz_convert("America/New_York").dt.date
    df = df.sort_values(["symbol", "ts_utc"]).copy()

    groups = df.groupby(["symbol", "trading_date"], sort=False)
    features = add_normalized_features(df, groups)

    # Target (Y): 15-minute ahead return
    df["future_close"] = groups["close"].shift(-3)
    df["future_time"] = groups["ts_utc"].shift(-3)
    df["future_return"] = ((df["future_close"] - df["close"]) / df["close"]) * 100

    valid_prev = (df["ts_utc"] - groups["ts_utc"].shift(5)) == pd.Timedelta(minutes=25)
    valid_fut = (df["future_time"] - df["ts_utc"]) == pd.Timedelta(minutes=15)

    df.loc[~valid_prev, features] = np.nan
    df.loc[~valid_fut, "future_return"] = np.nan
    df = df.dropna(subset=features + ["future_return"]).copy()

    train = df[(df["ts_utc"] >= "2024-01-01") & (df["ts_utc"] < "2025-01-01")]
    test = df[(df["ts_utc"] >= "2025-01-01") & (df["ts_utc"] < "2025-07-01")]

    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=42)
    tree.fit(train[features], train["future_return"])

    preds = tree.predict(test[features])
    print("=== DECISION TREE TRAINED ===")
    print(f"Test MAE: {mean_absolute_error(test['future_return'], preds):.6f}%")
    print(f"Test R2:  {r2_score(test['future_return'], preds):.6f}")

    return tree, features


# --------------------------------------------------
# 2. SINGLE CHECK CYCLE
# --------------------------------------------------

def check_market_and_trade(trading_client, data_client, tree, features, active_bot_trades):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] Checking market status...")

    clock = trading_client.get_clock()
    if not clock.is_open:
        print(f"Market is currently CLOSED. Next open: {clock.next_open}")
        return

    now_utc = datetime.now(timezone.utc)

    # 1. Close any bot trades that have been open for >= 15 minutes
    if AUTO_SELL_AFTER_15M:
        for symbol in list(active_bot_trades.keys()):
            entry_time = active_bot_trades[symbol]
            if now_utc - entry_time >= timedelta(minutes=15):
                sell_order = MarketOrderRequest(
                    symbol=symbol,
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                submitted = trading_client.submit_order(sell_order)
                print(f"  -> 15 MIN ELAPSED: Sold 1 share of {symbol} (ID: {submitted.id})")
                del active_bot_trades[symbol]

    # 2. Fetch the latest 5-minute bars for all 3 symbols
    request_params = StockBarsRequest(
        symbol_or_symbols=SYMBOLS_TO_TRADE,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=now_utc - timedelta(days=3),
        end=now_utc,
        feed=DataFeed.IEX
    )

    all_bars = data_client.get_stock_bars(request_params).df.reset_index()

    for symbol in SYMBOLS_TO_TRADE:
        bars = all_bars[all_bars["symbol"] == symbol].copy()
        if len(bars) < 6:
            print(f"[{symbol}] Not enough recent bars.")
            continue

        recent_bars = bars.tail(6).copy()
        recent_bars["dummy_group"] = 1
        groups = recent_bars.groupby("dummy_group", sort=False)
        add_normalized_features(recent_bars, groups)

        latest_row = recent_bars.iloc[[-1]][features]
        latest_close = recent_bars.iloc[-1]["close"]
        predicted_return = tree.predict(latest_row)[0]

        print(f"[{symbol}] Close: ${latest_close:.2f} | Predicted 15m Return: {predicted_return:.4f}%")

        # Avoid double-buying a symbol the bot is already holding for a 15m cycle
        if predicted_return > BUY_THRESHOLD and symbol not in active_bot_trades:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            submitted = trading_client.submit_order(order)
            active_bot_trades[symbol] = now_utc
            print(f"  -> BUY ORDER SUBMITTED: 1 share of {symbol} (ID: {submitted.id})")


# --------------------------------------------------
# 3. MAIN AUTOMATED 5-MINUTE LOOP
# --------------------------------------------------

def run_automated_loop():
    trained_tree, feature_cols = build_and_train_tree()

    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    account = trading_client.get_account()
    print(f"\nConnected to Alpaca Paper Account! Buying Power: ${account.buying_power}")
    print("Starting automated 5-minute loop (Press Ctrl + C in terminal to stop)...\n")

    # Tracks {symbol: entry_timestamp} for bot-initiated trades (won't touch your manual shares)
    active_bot_trades = {}

    while True:
        try:
            check_market_and_trade(
                trading_client,
                data_client,
                trained_tree,
                feature_cols,
                active_bot_trades
            )
        except Exception as e:
            print(f"[WARNING] Error during cycle (will retry in 5 mins): {e}")

        print(f"Sleeping for {LOOP_SECONDS // 60} minutes...")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    run_automated_loop()