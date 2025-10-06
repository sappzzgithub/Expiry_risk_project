# src/forecasting.py

import os
import pandas as pd
from prophet import Prophet

def main(preprocessed_csv_path="data/processed/processed_data.csv",
         forecast_dir="forecasts/product_level",
         use_existing_forecast=True):
    """
    Generates product-level forecasts using Prophet.
    If 'all_products_forecast.csv' exists and use_existing_forecast=True, it will be reused.

    Args:
        preprocessed_csv_path (str): Path to preprocessed data CSV.
        forecast_dir (str): Directory to save individual and combined forecasts.
        use_existing_forecast (bool): If True, reuse existing combined forecast if present.
    """
    os.makedirs(forecast_dir, exist_ok=True)
    combined_forecast_path = os.path.join(forecast_dir, "all_products_forecast.csv")

    # ✅ Check if combined forecast already exists
    if use_existing_forecast and os.path.exists(combined_forecast_path):
        print(f"✅ Existing combined forecast found: {combined_forecast_path}")
        return combined_forecast_path

    # ✅ Load preprocessed data
    date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
    df = pd.read_csv(preprocessed_csv_path, parse_dates=date_cols)

    # Aggregate daily sales per product
    df_agg = df.groupby(["Product_Name", "Date_Received"]).agg({"Sales_Volume": "sum"}).reset_index()

    forecast_horizon = 30  # days
    all_forecasts = []

    for product, group in df_agg.groupby("Product_Name"):
        if len(group) < 5:
            print(f"⚠️ Skipping {product} (not enough data points)")
            continue

        ts = group.rename(columns={"Date_Received": "ds", "Sales_Volume": "y"})

        # Train Prophet model
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        model.fit(ts)

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon, freq="D")
        forecast = model.predict(future)

        # Save individual forecast CSV
        out_file = os.path.join(forecast_dir, f"{product.replace('/', '_')}_forecast.csv")
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_file, index=False)
        forecast["Product_Name"] = product
        all_forecasts.append(forecast[["ds", "yhat", "Product_Name"]])

        print(f"✅ Forecast saved for {product} → {out_file}")

    # Save combined forecast
    if all_forecasts:
        combined = pd.concat(all_forecasts)
        combined.to_csv(combined_forecast_path, index=False)
        print(f"\n🎯 Combined forecast saved → {combined_forecast_path}")
    else:
        print("\n⚠️ No forecasts generated. Not enough data per product.")

    return combined_forecast_path

# Optional: allow standalone execution for testing
if __name__ == "__main__":
    main()


# # src/forecasting.py

# import os
# import pandas as pd
# from prophet import Prophet

# def main():
#     PROCESSED_PATH = "data/processed/processed_data.csv"
#     FORECAST_PATH = "forecasts/product_level"
#     os.makedirs(FORECAST_PATH, exist_ok=True)

#     date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
#     df = pd.read_csv(PROCESSED_PATH, parse_dates=date_cols)

#     df_agg = (
#         df.groupby(["Product_Name", "Date_Received"])
#           .agg({"Sales_Volume": "sum"})
#           .reset_index()
#     )

#     forecast_horizon = 30  # days
#     all_forecasts = []

#     for product, group in df_agg.groupby("Product_Name"):
#         if len(group) < 5:
#             print(f"⚠️ Skipping {product} (not enough data points)")
#             continue

#         ts = group.rename(columns={"Date_Received": "ds", "Sales_Volume": "y"})

#         model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
#         model.fit(ts)

#         future = model.make_future_dataframe(periods=forecast_horizon, freq="D")
#         forecast = model.predict(future)

#         out_file = os.path.join(FORECAST_PATH, f"{product.replace('/', '_')}_forecast.csv")
#         forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_file, index=False)

#         print(f"✅ Forecast saved for {product} → {out_file}")

#         forecast["Product_Name"] = product
#         all_forecasts.append(forecast[["ds", "yhat", "Product_Name"]])

#     if all_forecasts:
#         combined = pd.concat(all_forecasts)
#         combined.to_csv(os.path.join(FORECAST_PATH, "all_products_forecast.csv"), index=False)
#         print(f"\n🎯 Combined forecast saved → {FORECAST_PATH}/all_products_forecast.csv")
#     else:
#         print("\n⚠️ No forecasts generated. Not enough data per product.")

# # Optional: allow standalone execution
# if __name__ == "__main__":
#     main()