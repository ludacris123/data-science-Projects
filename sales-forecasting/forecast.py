"""Forecast daily sales with calendar, lag, and rolling-window features."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
FEATURES = ["day_of_week", "month", "day_of_year", "trend", "lag_1", "lag_7", "rolling_7"]


def make_sales(days: int = 730) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    trend = np.linspace(0, 35, days)
    weekly = 22 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
    yearly = 18 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    promotion = rng.binomial(1, 0.12, days)
    sales = 180 + trend + weekly + yearly + promotion * 45 + rng.normal(0, 12, days)
    return pd.DataFrame({"date": dates, "sales": sales.clip(20).round(2), "promotion": promotion})


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear
    data["trend"] = np.arange(len(data))
    data["lag_1"] = data["sales"].shift(1)
    data["lag_7"] = data["sales"].shift(7)
    data["rolling_7"] = data["sales"].shift(1).rolling(7).mean()
    return data.dropna()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    history = make_sales()
    history.to_csv(OUTPUT_DIR / "sales_history.csv", index=False)
    model_data = add_features(history)

    split = len(model_data) - 60
    train, test = model_data.iloc[:split], model_data.iloc[split:]
    model = RandomForestRegressor(
        n_estimators=350, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(train[FEATURES], train["sales"])
    predicted = model.predict(test[FEATURES])

    metrics = {
        "mae": round(mean_absolute_error(test["sales"], predicted), 2),
        "rmse": round(mean_squared_error(test["sales"], predicted) ** 0.5, 2),
        "mape_percent": round(
            np.mean(np.abs((test["sales"].to_numpy() - predicted) / test["sales"].to_numpy())) * 100, 2
        ),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    working = history[["date", "sales"]].copy()
    forecasts = []
    for _ in range(30):
        next_date = working["date"].iloc[-1] + pd.Timedelta(days=1)
        row = pd.DataFrame({
            "date": [next_date],
            "sales": [np.nan],
        })
        working = pd.concat([working, row], ignore_index=True)
        i = len(working) - 1
        feature_row = pd.DataFrame([{
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "day_of_year": next_date.dayofyear,
            "trend": i,
            "lag_1": working.loc[i - 1, "sales"],
            "lag_7": working.loc[i - 7, "sales"],
            "rolling_7": working.loc[i - 7:i - 1, "sales"].mean(),
        }])
        value = float(model.predict(feature_row[FEATURES])[0])
        working.loc[i, "sales"] = value
        forecasts.append({"date": next_date, "forecast_sales": round(value, 2)})

    future = pd.DataFrame(forecasts)
    future.to_csv(OUTPUT_DIR / "future_30_day_forecast.csv", index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(history["date"].tail(120), history["sales"].tail(120), label="Historical")
    plt.plot(future["date"], future["forecast_sales"], label="30-day forecast", linewidth=2)
    plt.title("Daily Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast.png", dpi=160)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
