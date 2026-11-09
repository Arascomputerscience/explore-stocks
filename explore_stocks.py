
"""
IB Computer Science SL2
Stock Database Exploration and Linear Regression Model

Database: Classroom NAS
Table: bars_5m
Training: January-December 2024
Testing: January-June 2025
Prediction: Stock percentage return 15 minutes ahead

READ-ONLY: The classroom database is never modified.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# DATABASE SETTINGS
# --------------------------------------------------

DB_PATH = Path(r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite")

DB_URI = (
    "file:////192.168.1.95/CS-CLASSROOM/"
    "intraday_5m.sqlite?mode=ro&immutable=1"
)

STOCK_TO_EXPLORE = "AAPL"
START_DATE = "2025-01-01"
END_DATE = "2025-02-01"
PREVIEW_ROWS = 15


# --------------------------------------------------
# SQL DISPLAY FUNCTIONS
# --------------------------------------------------

def show_rows(title, columns, rows):
    print(f"\n=== {title} ===")
    print(" | ".join(columns))

    for row in rows:
        print(" | ".join(str(value) for value in row))


def query(conn, title, sql, params=()):
    cursor = conn.execute(sql, params)

    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()

    show_rows(title, columns, rows)

    return rows


# --------------------------------------------------
# LINEAR REGRESSION MODEL
# --------------------------------------------------

def train_regression_model(conn):

    print("\n" + "=" * 55)
    print("LINEAR REGRESSION TRADING MODEL")
    print("=" * 55)

    # Retrieve historical stock data
    sql = """
        SELECT
            symbol,
            ts_utc,
            open,
            high,
            low,
            close,
            volume
        FROM bars_5m
        WHERE symbol IN ('AAPL', 'MSFT', 'NVDA')
          AND ts_utc >= '2024-01-01'
          AND ts_utc < '2025-07-01'
        ORDER BY symbol, ts_utc
    """

    df = pd.read_sql_query(sql, conn)

    print("\nTotal stock records retrieved:", len(df))

    if df.empty:
        raise ValueError("No stock records were retrieved.")

    # Convert timestamps
    df["ts_utc"] = pd.to_datetime(
        df["ts_utc"], utc=True
    )

    # Convert timestamps to New York trading dates
    df["trading_date"] = (
        df["ts_utc"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    # Sort stock records
    df = df.sort_values(
        ["symbol", "ts_utc"]
    ).copy()

    # Group by stock and trading day
    groups = df.groupby(
        ["symbol", "trading_date"],
        sort=False
    )

    # ----------------------------------------------
    # INPUT VARIABLES (X)
    # ----------------------------------------------

    # Input 1: Current closing price
    df["current_close"] = df["close"]

    # Input 2: Recent percentage return
    df["previous_time"] = groups["ts_utc"].shift(1)

    df["recent_return"] = (
        groups["close"].pct_change() * 100
    )

    # Input 3: High-low price range
    df["high_low_range"] = (
        df["high"] - df["low"]
    )

    # Input 4: Trading volume
    df["trading_volume"] = df["volume"]

    # ----------------------------------------------
    # TARGET VARIABLE (Y)
    # ----------------------------------------------

    # Closing price three five-minute bars ahead
    df["future_close"] = groups["close"].shift(-3)

    # Timestamp three bars ahead
    df["future_time"] = groups["ts_utc"].shift(-3)

    # Calculate future percentage return
    df["future_return"] = (
        (df["future_close"] - df["close"])
        / df["close"]
    ) * 100

    # ----------------------------------------------
    # DATA VALIDATION
    # ----------------------------------------------

    # Previous bar must be exactly five minutes earlier
    valid_previous = (
        df["ts_utc"] - df["previous_time"]
        == pd.Timedelta(minutes=5)
    )

    # Future bar must be exactly fifteen minutes later
    valid_future = (
        df["future_time"] - df["ts_utc"]
        == pd.Timedelta(minutes=15)
    )

    df.loc[~valid_previous, "recent_return"] = np.nan
    df.loc[~valid_future, "future_return"] = np.nan

    # Remove incomplete records
    df = df.dropna(
        subset=[
            "current_close",
            "recent_return",
            "high_low_range",
            "trading_volume",
            "future_return"
        ]
    ).copy()

    # ----------------------------------------------
    # TRAINING AND TESTING PERIODS
    # ----------------------------------------------

    train = df[
        (df["ts_utc"] >= "2024-01-01")
        & (df["ts_utc"] < "2025-01-01")
    ].copy()

    test = df[
        (df["ts_utc"] >= "2025-01-01")
        & (df["ts_utc"] < "2025-07-01")
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing data is empty. "
            "Check the available historical dates."
        )

    print("\n=== DATASET INFORMATION ===")

    print("Training records:", len(train))
    print("Testing records:", len(test))

    print("\nTraining records by stock:")
    print(train.groupby("symbol").size())

    print("\nTesting records by stock:")
    print(test.groupby("symbol").size())

    # ----------------------------------------------
    # SELECT INPUTS AND TARGET
    # ----------------------------------------------

    features = [
        "current_close",
        "recent_return",
        "high_low_range",
        "trading_volume"
    ]

    X_train = train[features]
    y_train = train["future_return"]

    X_test = test[features]
    y_test = test["future_return"]

    # ----------------------------------------------
    # TRAIN LINEAR REGRESSION
    # ----------------------------------------------

    print("\n=== TRAINING MODEL ===")

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    print("Linear regression model trained successfully!")

    # ----------------------------------------------
    # MAKE PREDICTIONS
    # ----------------------------------------------

    predictions = model.predict(X_test)

    # ----------------------------------------------
    # EVALUATE MODEL
    # ----------------------------------------------

    mae = mean_absolute_error(
        y_test, predictions
    )

    mse = mean_squared_error(
        y_test, predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test, predictions
    )

    print("\n=== MODEL EVALUATION ===")

    print(f"MAE: {mae:.6f} percentage points")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f} percentage points")
    print(f"R-squared: {r2:.6f}")

    # ----------------------------------------------
    # COMPARE WITH SIMPLE MEAN BASELINE
    # ----------------------------------------------

    mean_predictions = np.full(
        len(y_test),
        y_train.mean()
    )

    mean_mae = mean_absolute_error(
        y_test,
        mean_predictions
    )

    print("\n=== MEAN BASELINE COMPARISON ===")

    print(f"Mean baseline MAE: {mean_mae:.6f}")
    print(f"Regression MAE: {mae:.6f}")

    if mae < mean_mae:
        print("Regression has lower MAE than the mean baseline.")
    else:
        print("Regression does not improve on the mean baseline MAE.")

    # ----------------------------------------------
    # MODEL COEFFICIENTS
    # ----------------------------------------------

    print("\n=== MODEL COEFFICIENTS ===")

    for feature, coefficient in zip(
        features,
        model.coef_
    ):
        print(f"{feature}: {coefficient:.8f}")

    print(f"Intercept: {model.intercept_:.8f}")

    # ----------------------------------------------
    # SAMPLE PREDICTIONS
    # ----------------------------------------------

    results = test[
        ["symbol", "ts_utc"]
    ].copy()

    results["actual_return"] = y_test
    results["predicted_return"] = predictions

    print("\n=== SAMPLE PREDICTIONS ===")

    print(
        results.head(15).to_string(index=False)
    )

    # ----------------------------------------------
    # RESULTS BY STOCK
    # ----------------------------------------------

    print("\n=== RESULTS BY STOCK ===")

    results["absolute_error"] = abs(
        results["actual_return"]
        - results["predicted_return"]
    )

    print(
        results.groupby("symbol")[
            "absolute_error"
        ].mean()
    )

    print("\nModel evaluation complete.")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot see {DB_PATH}. "
            "Connect to the NAS using the CS-Students account."
        )

    # Connect to classroom database in read-only mode
    with sqlite3.connect(
        DB_URI,
        uri=True
    ) as conn:

        conn.execute("PRAGMA query_only = ON")

        # 1. Database tables
        query(conn, "Database tables", """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
        """)

        # 2. Table schema
        query(
            conn,
            "bars_5m column names and types",
            "PRAGMA table_info(bars_5m)"
        )

        # 3. Available stock symbols
        query(conn, "Available stock symbols", """
            SELECT DISTINCT symbol
            FROM bars_5m
            ORDER BY symbol
        """)

        # 4. Historical coverage
        query(conn, "History available by stock", """
            SELECT
                symbol,
                COUNT(*) AS bar_count,
                MIN(ts_utc) AS first_bar,
                MAX(ts_utc) AS last_bar
            FROM bars_5m
            GROUP BY symbol
            ORDER BY bar_count DESC
        """)

        # 5. Recent five-minute bars
        query(
            conn,
            f"Recent five-minute bars for {STOCK_TO_EXPLORE}",
            """
            SELECT
                symbol, ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ?
            ORDER BY ts_utc DESC
            LIMIT ?
            """,
            (STOCK_TO_EXPLORE, PREVIEW_ROWS)
        )

        # 6. Chosen date range
        query(
            conn,
            f"{STOCK_TO_EXPLORE}: chosen date range",
            """
            SELECT
                symbol, ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ?
              AND ts_utc >= ?
              AND ts_utc < ?
            ORDER BY ts_utc ASC
            LIMIT ?
            """,
            (
                STOCK_TO_EXPLORE,
                START_DATE,
                END_DATE,
                PREVIEW_ROWS
            )
        )

        # 7. Date range summary
        query(
            conn,
            f"{STOCK_TO_EXPLORE}: period summary",
            """
            SELECT
                symbol,
                COUNT(*) AS bar_count,
                MIN(ts_utc) AS first_bar,
                MAX(ts_utc) AS last_bar,
                MIN(low) AS lowest_price,
                MAX(high) AS highest_price,
                AVG(volume) AS average_bar_volume
            FROM bars_5m
            WHERE symbol = ?
              AND ts_utc >= ?
              AND ts_utc < ?
            GROUP BY symbol
            """,
            (
                STOCK_TO_EXPLORE,
                START_DATE,
                END_DATE
            )
        )

        # 8. Train and evaluate linear regression
        train_regression_model(conn)

    print("\n" + "=" * 55)
    print("EXPLORATION AND MODEL TRAINING COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
