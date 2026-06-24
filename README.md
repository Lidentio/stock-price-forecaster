# 📈 stock-price-forecaster

An end-to-end time-series forecasting pipeline for stock prices — data cleaning, feature engineering, multi-model training, accuracy evaluation, and future price projection with confidence intervals.

---

## Features

- **Realistic data generation** — ships with a synthetic 4-year OHLCV dataset so you can run it immediately; swap in your own CSV at any time
- **Automated preprocessing** — fills missing trading days, interpolates NaNs, flags outliers via rolling z-score
- **Feature engineering** — lag features, moving averages (7/30-day), rolling volatility, day-of-week, and Fourier seasonality terms
- **Four forecasting models** trained and compared head-to-head:
  - SARIMA (classical time-series)
  - Linear Regression (engineered features)
  - Random Forest (non-linear, tree-based)
  - Naïve baseline (last observed value)
- **Four accuracy metrics** — MAE, RMSE, MAPE, R²
- **Future forecast** — projects prices N trading days ahead with a 95% confidence band
- **Five auto-generated charts** saved to `outputs/`

---

## Quickstart

```bash
git clone https://github.com/your-username/stock-price-forecaster.git
cd stock-price-forecaster
pip install -r requirements.txt
python stock_price_forecast.py
```

Results and charts are written to `outputs/`. On first run the script generates and saves `data/sample_stock_data.csv` automatically.

---

## Using Your Own Data

Drop a CSV with the following columns into the `data/` folder and point `DATA_PATH` at it:

```
Date, Open, High, Low, Close, Volume
2021-01-04, 154.21, 156.20, 152.92, 153.71, 961259
...
```

The script handles messy real-world data — gaps, missing values, and non-trading days are all handled during preprocessing.

---

## Project Structure

```
stock-price-forecaster/
│
├── stock_price_forecast.py   # Main pipeline script
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_stock_data.csv # Auto-generated on first run
│
└── outputs/
    ├── 01_historical_overview.png
    ├── 02_model_comparison.png
    ├── 03_residuals.png
    ├── 04_future_forecast.png
    ├── 05_metrics_table.png
    ├── model_evaluation_metrics.csv
    └── future_forecast.csv
```

---

## Results (Sample Dataset)

Evaluated on a 60-day chronological holdout (Jan–Mar 2025):

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Linear Regression** | **0.80** | **0.99** | **0.39%** | **0.955** |
| SARIMA | 6.62 | 8.00 | 3.16% | -1.96 |
| Naïve baseline | 6.64 | 8.02 | 3.17% | -1.98 |
| Random Forest | 7.19 | 8.59 | 3.43% | -2.42 |

Linear Regression wins on this dataset because `lag_1` (yesterday's close) is a strong one-step predictor on a trending series. SARIMA and Random Forest both struggle to extrapolate beyond the price range seen during training — a common pattern with real stock data and a key reason to always compare multiple approaches.

---

## Configuration

All key parameters are at the top of `stock_price_forecast.py`:

```python
DATA_PATH      = "data/sample_stock_data.csv"  # path to your CSV
TEST_SIZE      = 60     # trading days held out for evaluation
FUTURE_HORIZON = 30     # trading days to forecast beyond known data
RANDOM_SEED    = 42
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
statsmodels
```

Install with:

```bash
pip install -r requirements.txt
```

---

## License

MIT
