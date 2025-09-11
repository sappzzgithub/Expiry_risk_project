# src/forecasting.py
"""
Product-Level Forecasting Module
--------------------------------
This module predicts future demand (Sales_Volume) for each Product_Name.
Steps:
1. Load processed dataset
2. Aggregate Sales_Volume by Product_Name & Date_Received
3. Fit Prophet model for each Product_Name
4. Save forecast results in forecasts/ folder
"""

import os
import pandas as pd
from prophet import Prophet

# -------------------------
# Paths
# -------------------------
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/processed_data.csv"
FORECAST_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/forecasts/product_level"
os.makedirs(FORECAST_PATH, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
df = pd.read_csv(PROCESSED_PATH, parse_dates=date_cols)

# -------------------------
# Aggregate Sales by Product_Name & Date
# -------------------------
df_agg = (
    df.groupby(["Product_Name", "Date_Received"])
      .agg({"Sales_Volume": "sum"})
      .reset_index()
)

# -------------------------
# Forecast per Product
# -------------------------
forecast_horizon = 30  # days
all_forecasts = []

for product, group in df_agg.groupby("Product_Name"):
    if len(group) < 5:
        print(f"⚠️ Skipping {product} (not enough data points)")
        continue

    # Prophet expects columns: ds, y
    ts = group.rename(columns={"Date_Received": "ds", "Sales_Volume": "y"})

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.fit(ts)

    future = model.make_future_dataframe(periods=forecast_horizon, freq="D")
    forecast = model.predict(future)

    # Save individual forecast
    out_file = os.path.join(FORECAST_PATH, f"{product.replace('/', '_')}_forecast.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_file, index=False)

    print(f"✅ Forecast saved for {product} → {out_file}")

    forecast["Product_Name"] = product
    all_forecasts.append(forecast[["ds", "yhat", "Product_Name"]])

# -------------------------
# Combine all forecasts
# -------------------------
if all_forecasts:
    combined = pd.concat(all_forecasts)
    combined.to_csv(os.path.join(FORECAST_PATH, "all_products_forecast.csv"), index=False)
    print(f"\n🎯 Combined forecast saved → {FORECAST_PATH}/all_products_forecast.csv")
else:
    print("\n⚠️ No forecasts generated. Not enough data per product.")