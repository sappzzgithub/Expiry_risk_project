# src/risk_scoring.py
"""
Risk Scoring Module (3 Categories)
----------------------------------
- Flags expired products separately
- Assigns High (overstock) or Low (understock) risk for non-expired items
- Saves results to data/external/risk_scores.csv
"""

import os
import pandas as pd

# -------------------------
# Paths
# -------------------------
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/processed_data.csv"
FORECAST_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/forecasts/product_level/all_products_forecast.csv"
OUTPUT_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/external/risk_scores.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(PROCESSED_PATH, parse_dates=["Date_Received", "Last_Order_Date", "Expiration_Date"])
forecast = pd.read_csv(FORECAST_PATH, parse_dates=["ds"])

# -------------------------
# Merge forecast into main df
# -------------------------
# Use the latest forecast for each product
latest_forecast = forecast.groupby("Product_Name")["yhat"].last().reset_index()
latest_forecast.rename(columns={"yhat": "Forecasted_Demand"}, inplace=True)

df = df.merge(latest_forecast, on="Product_Name", how="left")

# -------------------------
# Risk scoring (3 categories)
# -------------------------
def assign_risk(row):
    # Expired first
    if row["Expiry_Class"] == "Expired":
        return "Expired"
    # Overstock → forecast < stock
    elif row["Forecasted_Demand"] < row["Stock_Quantity"]:
        return "High"
    # Understock → forecast >= stock
    else:
        return "Low"

df["Risk_Level"] = df.apply(assign_risk, axis=1)

# -------------------------
# Save results
# -------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Risk scoring complete. Results saved → {OUTPUT_PATH}")

# Summary
print("\nRisk Level Distribution:")
print(df["Risk_Level"].value_counts())