# src/forecasting.py
"""
Forecasting Module (Category-level)
-----------------------------------
This module predicts future demand (Sales_Volume) using category-level aggregation.
Instead of Product_ID (which has too few records), we forecast demand for each Category.

Steps:
1. Load processed dataset
2. Aggregate Sales_Volume by Category & Date_Received
3. Fit Prophet model for each Category
4. Save forecast results in forecasts/ folder
"""

import os
import pandas as pd
from prophet import Prophet

# -------------------------
# Paths
# -------------------------
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/intermediate_data.csv"
FORECAST_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/forecasts"
os.makedirs(FORECAST_PATH, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
df = pd.read_csv(PROCESSED_PATH, parse_dates=date_cols)

# -------------------------
# Aggregate Sales by Category & Date
# -------------------------
# We’ll use Date_Received as a proxy for sales date (since daily logs are missing)
df_agg = (
    df.groupby(["Category", "Date_Received"])
      .agg({"Sales_Volume": "sum"})
      .reset_index()
)

# -------------------------
# Forecast per Category
# -------------------------
forecast_horizon = 30  # days
all_forecasts = []

for category, group in df_agg.groupby("Category"):
    if len(group) < 5:
        print(f"⚠️ Skipping {category} (not enough data points)")
        continue

    # Prophet expects columns: ds, y
    ts = group.rename(columns={"Date_Received": "ds", "Sales_Volume": "y"})

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.fit(ts)

    future = model.make_future_dataframe(periods=forecast_horizon, freq="D")
    forecast = model.predict(future)

    # Save forecast
    out_file = os.path.join(FORECAST_PATH, f"{category}_forecast.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_file, index=False)

    print(f"✅ Forecast saved for {category} → {out_file}")

    forecast["Category"] = category
    all_forecasts.append(forecast[["ds", "yhat", "Category"]])

# -------------------------
# Combine all forecasts
# -------------------------
if all_forecasts:
    combined = pd.concat(all_forecasts)
    combined.to_csv(os.path.join(FORECAST_PATH, "all_categories_forecast.csv"), index=False)
    print(f"\n🎯 Combined forecast saved → {FORECAST_PATH}/all_categories_forecast.csv")
else:
    print("\n⚠️ No forecasts generated. Not enough data per category.")
