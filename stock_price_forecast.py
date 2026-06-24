"""
Predictive Analytics for Stock Prices — Time-Series Forecasting
==================================================================

End-to-end pipeline that:
  1. Loads historical price data (or generates a realistic sample dataset
     if none is provided)
  2. Cleans and preprocesses it (missing dates, NaNs, outliers)
  3. Engineers time-series features (lags, moving averages, seasonality)
  4. Trains and compares four forecasting approaches:
       - SARIMA            (classic time-series model)
       - Linear Regression (regression on engineered features)
       - Random Forest     (non-linear regression, for comparison)
       - Naive baseline    (last observed value — sanity check)
  5. Evaluates accuracy (MAE, RMSE, MAPE, R²)
  6. Visualizes historical data, predictions, residuals, and a future forecast

Usage
-----
    python stock_price_forecast.py

To use your OWN data instead of the bundled sample: drop a CSV with columns
[Date, Open, High, Low, Close, Volume] at DATA_PATH below (or edit DATA_PATH)
and the script will use it automatically instead of generating synthetic data.

Dependencies: numpy, pandas, matplotlib, scikit-learn, statsmodels
    pip install numpy pandas matplotlib scikit-learn statsmodels
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_PATH = "data/sample_stock_data.csv"   # used if it exists; else generated
OUTPUT_DIR = "outputs"
TEST_SIZE = 60          # trading days held out for evaluation
FUTURE_HORIZON = 30     # trading days to forecast beyond the known data
RANDOM_SEED = 42

COLORS = {
    "actual": "#1f3a5f",
    "train": "#9aa5b1",
    "sarima": "#d9534f",
    "linreg": "#2e8b57",
    "rf": "#c08400",
    "naive": "#9b59b6",
    "ci": "#d9534f",
}


# ----------------------------------------------------------------------------
# 1. DATA LOADING / GENERATION
# ----------------------------------------------------------------------------
def generate_synthetic_stock_data(start="2021-01-01", periods=1095, seed=RANDOM_SEED):
    """
    Creates a realistic synthetic daily stock-price series with:
      - an exponential growth trend
      - yearly + quarterly seasonality
      - a volatility-clustered random walk component
      - a handful of missing trading days and NaNs (real-world messiness)
    This stands in for a real OHLCV dataset so the pipeline can be demoed
    end-to-end without needing a live market data feed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    n = len(dates)
    t = np.arange(n)

    base_price = 150.0
    drift = 0.00035
    trend = base_price * np.exp(drift * t)

    yearly = 8 * np.sin(2 * np.pi * t / 252)
    quarterly = 3 * np.sin(2 * np.pi * t / 63 + 1.0)

    vol_regime = 1 + 0.5 * (np.sin(2 * np.pi * t / 400) > 0.3)
    shocks = rng.normal(0, 1.1, n) * vol_regime
    random_walk = np.cumsum(shocks) * 0.4

    noise = rng.normal(0, 0.8, n)
    close = np.maximum(trend + yearly + quarterly + random_walk + noise, 1.0)

    daily_range = np.abs(rng.normal(1.5, 0.5, n)) + 0.5
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.5, 0.4, n)) + daily_range * 0.3
    low = np.minimum(open_, close) - np.abs(rng.normal(0.5, 0.4, n)) - daily_range * 0.3
    volume = np.maximum(
        (1_000_000 + rng.normal(0, 150_000, n) + np.abs(shocks) * 200_000).astype(int),
        50_000,
    )

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )

    # Introduce realistic messiness: a few dropped trading days and stray NaNs
    missing_idx = rng.choice(n, size=max(3, n // 200), replace=False)
    df = df.drop(index=missing_idx).reset_index(drop=True)
    nan_idx = rng.choice(len(df), size=5, replace=False)
    df.loc[nan_idx, "Close"] = np.nan

    return df


def load_data():
    """Loads DATA_PATH if it exists; otherwise generates and saves sample data."""
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("No dataset found — generating a realistic synthetic sample dataset.")
        df = generate_synthetic_stock_data()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Sample dataset saved to {DATA_PATH}")
    return df


# ----------------------------------------------------------------------------
# 2. CLEANING & PREPROCESSING
# ----------------------------------------------------------------------------
def clean_and_preprocess(df):
    """
    - Parses dates, removes duplicates
    - Reindexes onto a full business-day calendar to expose gaps
    - Fills missing values via linear interpolation
    - Flags (but does not silently drop) extreme outliers via rolling z-score
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
    df = df.set_index("Date")

    full_idx = pd.bdate_range(df.index.min(), df.index.max())
    n_gaps = len(full_idx) - len(df)
    df = df.reindex(full_idx)
    df.index.name = "Date"

    n_missing = df["Close"].isna().sum()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].interpolate(method="linear").ffill().bfill()

    roll_mean = df["Close"].rolling(30, min_periods=5).mean()
    roll_std = df["Close"].rolling(30, min_periods=5).std()
    z_score = (df["Close"] - roll_mean) / roll_std
    n_outliers = int((z_score.abs() > 4).sum())

    print(f"  Filled {n_gaps} missing trading-day rows")
    print(f"  Interpolated {n_missing} missing Close values")
    print(f"  Flagged {n_outliers} potential outliers (|z| > 4, none removed)")

    return df


# ----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def engineer_features(df):
    """Builds lag, moving-average, and seasonal features for the regression models."""
    feat = df.copy()
    feat["trend_idx"] = np.arange(len(feat))
    feat["day_of_week"] = feat.index.dayofweek
    feat["day_of_year"] = feat.index.dayofyear
    feat["sin_year"] = np.sin(2 * np.pi * feat["day_of_year"] / 365.25)
    feat["cos_year"] = np.cos(2 * np.pi * feat["day_of_year"] / 365.25)
    feat["ma_7"] = feat["Close"].rolling(7).mean()
    feat["ma_30"] = feat["Close"].rolling(30).mean()
    feat["volatility_30"] = feat["Close"].rolling(30).std()
    for lag in (1, 2, 3, 5, 7):
        feat[f"lag_{lag}"] = feat["Close"].shift(lag)
    return feat.dropna()


FEATURE_COLS = [
    "trend_idx", "day_of_week", "sin_year", "cos_year",
    "ma_7", "ma_30", "volatility_30",
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_7",
]


# ----------------------------------------------------------------------------
# 4. MODELING
# ----------------------------------------------------------------------------
def train_and_predict(df_clean, feat):
    """Trains all four models on a chronological train/test split and returns predictions."""
    train_feat, test_feat = feat.iloc[:-TEST_SIZE], feat.iloc[-TEST_SIZE:]
    y_train, y_test = train_feat["Close"], test_feat["Close"]

    predictions, models = {}, {}

    # --- Naive baseline: tomorrow = today -----------------------------------
    predictions["Naive (last value)"] = np.full(len(y_test), y_train.iloc[-1])

    # --- Linear Regression on engineered features ---------------------------
    lin_reg = LinearRegression().fit(train_feat[FEATURE_COLS], y_train)
    predictions["Linear Regression"] = lin_reg.predict(test_feat[FEATURE_COLS])
    models["Linear Regression"] = lin_reg

    # --- Random Forest on the same features ----------------------------------
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_SEED)
    rf.fit(train_feat[FEATURE_COLS], y_train)
    predictions["Random Forest"] = rf.predict(test_feat[FEATURE_COLS])
    models["Random Forest"] = rf

    # --- SARIMA on the raw Close series --------------------------------------
    sarima_train = df_clean["Close"].iloc[:-TEST_SIZE]
    sarima_model = SARIMAX(
        sarima_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 5),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    sarima_res = sarima_model.fit(disp=False)
    sarima_fc = sarima_res.get_forecast(steps=TEST_SIZE)
    predictions["SARIMA"] = sarima_fc.predicted_mean.values
    models["SARIMA"] = sarima_res

    return predictions, models, test_feat.index, y_test


def evaluate(predictions, y_test):
    """Computes MAE, RMSE, MAPE, R² for each model and returns a results DataFrame."""
    rows = []
    for name, pred in predictions.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100
        r2 = r2_score(y_test, pred)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def forecast_future(df_clean):
    """Refits SARIMA on the FULL cleaned series and forecasts FUTURE_HORIZON days ahead."""
    full_model = SARIMAX(
        df_clean["Close"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 5),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    full_res = full_model.fit(disp=False)
    fc = full_res.get_forecast(steps=FUTURE_HORIZON)
    future_dates = pd.bdate_range(
        df_clean.index[-1] + pd.Timedelta(days=1), periods=FUTURE_HORIZON
    )
    mean = pd.Series(fc.predicted_mean.values, index=future_dates)
    ci = fc.conf_int(alpha=0.05)
    ci.index = future_dates
    return mean, ci


# ----------------------------------------------------------------------------
# 5. VISUALIZATION
# ----------------------------------------------------------------------------
def plot_overview(df_clean, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    split_date = df_clean.index[-TEST_SIZE]
    ax.plot(df_clean.index[df_clean.index < split_date],
            df_clean["Close"][df_clean.index < split_date],
            color=COLORS["train"], lw=1.2, label="Training period")
    ax.plot(df_clean.index[df_clean.index >= split_date],
            df_clean["Close"][df_clean.index >= split_date],
            color=COLORS["actual"], lw=1.4, label="Held-out test period")
    ax.axvline(split_date, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Historical Closing Price — Train / Test Split", fontsize=13, weight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/01_historical_overview.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(test_idx, y_test, predictions, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_idx, y_test.values, color=COLORS["actual"], lw=2.2, label="Actual", zorder=5)
    style_map = {
        "SARIMA": COLORS["sarima"],
        "Linear Regression": COLORS["linreg"],
        "Random Forest": COLORS["rf"],
        "Naive (last value)": COLORS["naive"],
    }
    for name, pred in predictions.items():
        ax.plot(test_idx, pred, color=style_map[name], lw=1.4, ls="--", label=name, alpha=0.9)
    ax.set_title("Model Predictions vs. Actual Price (Test Period)", fontsize=13, weight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/02_model_comparison.png", dpi=150)
    plt.close(fig)


def plot_residuals(test_idx, y_test, predictions, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    style_map = {
        "SARIMA": COLORS["sarima"], "Linear Regression": COLORS["linreg"],
        "Random Forest": COLORS["rf"], "Naive (last value)": COLORS["naive"],
    }
    for ax, (name, pred) in zip(axes.flat, predictions.items()):
        resid = y_test.values - pred
        ax.bar(test_idx, resid, color=style_map[name], width=1.0, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{name} — Residuals", fontsize=10, weight="bold")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Prediction Residuals by Model (Actual − Predicted)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/03_residuals.png", dpi=150)
    plt.close(fig)


def plot_future_forecast(df_clean, future_mean, future_ci, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    history = df_clean["Close"].iloc[-150:]
    ax.plot(history.index, history.values, color=COLORS["actual"], lw=1.6, label="Historical")
    ax.plot(future_mean.index, future_mean.values, color=COLORS["sarima"], lw=1.8,
            ls="--", label=f"SARIMA forecast (+{FUTURE_HORIZON}d)")
    ax.fill_between(future_mean.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                     color=COLORS["ci"], alpha=0.15, label="95% confidence interval")
    ax.axvline(df_clean.index[-1], color="black", lw=0.8, ls=":", alpha=0.6)
    ax.set_title(f"{FUTURE_HORIZON}-Day Future Price Forecast (SARIMA)", fontsize=13, weight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/04_future_forecast.png", dpi=150)
    plt.close(fig)


def plot_metrics_table(results_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(results_df)))
    ax.axis("off")
    display_df = results_df.copy()
    for col in ["MAE", "RMSE", "MAPE (%)", "R2"]:
        display_df[col] = display_df[col].round(3)
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor("#1f3a5f")
        table[0, j].set_text_props(color="white", weight="bold")
    for i in range(len(display_df)):
        color = "#eaf3ea" if i == 0 else "white"
        for j in range(len(display_df.columns)):
            table[i + 1, j].set_facecolor(color)
    ax.set_title("Model Accuracy Comparison (Test Period)", fontsize=13, weight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/05_metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1 — Loading data")
    print("=" * 70)
    raw_df = load_data()
    print(f"  Rows: {len(raw_df)}  |  Date range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")

    print("\n" + "=" * 70)
    print("STEP 2 — Cleaning & preprocessing")
    print("=" * 70)
    df_clean = clean_and_preprocess(raw_df)

    print("\n" + "=" * 70)
    print("STEP 3 — Feature engineering")
    print("=" * 70)
    feat = engineer_features(df_clean)
    print(f"  Built {len(FEATURE_COLS)} features across {len(feat)} usable rows")

    print("\n" + "=" * 70)
    print(f"STEP 4 — Training models (train/test split, {TEST_SIZE}-day holdout)")
    print("=" * 70)
    predictions, models, test_idx, y_test = train_and_predict(df_clean, feat)
    print("  Trained: Naive baseline, Linear Regression, Random Forest, SARIMA")

    print("\n" + "=" * 70)
    print("STEP 5 — Evaluating accuracy")
    print("=" * 70)
    results_df = evaluate(predictions, y_test)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{OUTPUT_DIR}/model_evaluation_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print(f"STEP 6 — Forecasting {FUTURE_HORIZON} days into the future (SARIMA)")
    print("=" * 70)
    future_mean, future_ci = forecast_future(df_clean)
    future_out = pd.DataFrame({
        "forecast": future_mean,
        "lower_95": future_ci.iloc[:, 0].values,
        "upper_95": future_ci.iloc[:, 1].values,
    })
    future_out.to_csv(f"{OUTPUT_DIR}/future_forecast.csv", index_label="Date")
    print(future_out.head())

    print("\n" + "=" * 70)
    print("STEP 7 — Generating visualizations")
    print("=" * 70)
    plot_overview(df_clean, OUTPUT_DIR)
    plot_model_comparison(test_idx, y_test, predictions, OUTPUT_DIR)
    plot_residuals(test_idx, y_test, predictions, OUTPUT_DIR)
    plot_future_forecast(df_clean, future_mean, future_ci, OUTPUT_DIR)
    plot_metrics_table(results_df, OUTPUT_DIR)
    print(f"  Charts saved to ./{OUTPUT_DIR}/")

    best_model = results_df.iloc[0]["Model"]
    print("\n" + "=" * 70)
    print(f"DONE. Best performing model on held-out test data: {best_model}")
    print("=" * 70)


if __name__ == "__main__":
    main()
