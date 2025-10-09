# src/risk_scoring.py

import os
import pandas as pd
import joblib

def main(preprocessed_csv_path="data/processed/processed_data.csv",
         forecast_path="forecasts/product_level/all_products_forecast.csv",
         model_path="models/best_model.pkl",
         label_encoder_path="models/label_encoder.pkl",
         output_path="data/external/risk_scores.csv"):
    """
    Generates risk scores for inventory using expiry predictions and forecasted demand.

    Args:
        preprocessed_csv_path (str): Path to preprocessed inventory CSV.
        forecast_path (str): Path to combined forecast CSV.
        model_path (str): Path to saved classifier for Expiry_Class prediction.
        label_encoder_path (str): Path to LabelEncoder for Expiry_Class.
        output_path (str): Path to save risk scores CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ Load preprocessed inventory
    df = pd.read_csv(preprocessed_csv_path, parse_dates=["Date_Received", "Last_Order_Date", "Expiration_Date"])

    # Validate required columns before parsing dates
    required_columns = ["Date_Received", "Expiration_Date", "Last_Order_Date"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in preprocessed data: {', '.join(missing_columns)}")

    # ✅ Predict Expiry_Class if not present
    if "Expiry_Class" not in df.columns:
        print("⚡ Expiry_Class not found in data → Using saved model for prediction")
        X_features = [
            "Category", "Shelf_Life", "Stock_Quantity", "Stock_Value",
            "Sales_Volume", "Inventory_Turnover_Rate", "Unit_Price",
            "Days_Until_Expiry", "Remaining_Shelf_Life_Ratio", "Stock_Age"
        ]
        X = pd.get_dummies(df[X_features], drop_first=True)

        # Load model & label encoder
        model = joblib.load(model_path)
        le = joblib.load(label_encoder_path)

        df["Expiry_Class"] = le.inverse_transform(model.predict(X))

    # ✅ Load forecasted demand
    if os.path.exists(forecast_path):
        forecast = pd.read_csv(forecast_path, parse_dates=["ds"])
        latest_forecast = forecast.groupby("Product_Name")["yhat"].last().reset_index()
        latest_forecast.rename(columns={"yhat": "Forecasted_Demand"}, inplace=True)
        df = df.merge(latest_forecast, on="Product_Name", how="left")
    else:
        print(f"⚠️ Forecast file not found: {forecast_path}. Forecasted_Demand will be empty.")
        df["Forecasted_Demand"] = pd.NA

    # ✅ Assign Risk_Level
    def assign_risk(row):
        if row["Expiry_Class"] == "Expired":
            return "Expired"
        elif pd.notna(row["Forecasted_Demand"]) and row["Forecasted_Demand"] < row["Stock_Quantity"]:
            return "High"
        else:
            return "Low"

    df["Risk_Level"] = df.apply(assign_risk, axis=1)

    # ✅ Save risk scores
    df.to_csv(output_path, index=False)
    print(f"✅ Risk scoring complete. Results saved → {output_path}")

    print("\nRisk Level Distribution:")
    print(df["Risk_Level"].value_counts())

# Optional: allow standalone execution
if __name__ == "__main__":
    main()



# # src/risk_scoring.py

# import os
# import pandas as pd

# def main():
#     PROCESSED_PATH = "data/processed/processed_data.csv"
#     FORECAST_PATH = "forecasts/product_level/all_products_forecast.csv"
#     OUTPUT_PATH = "data/external/risk_scores.csv"
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

#     df = pd.read_csv(PROCESSED_PATH, parse_dates=["Date_Received", "Last_Order_Date", "Expiration_Date"])
#     forecast = pd.read_csv(FORECAST_PATH, parse_dates=["ds"])

#     latest_forecast = forecast.groupby("Product_Name")["yhat"].last().reset_index()
#     latest_forecast.rename(columns={"yhat": "Forecasted_Demand"}, inplace=True)

#     df = df.merge(latest_forecast, on="Product_Name", how="left")

#     def assign_risk(row):
#         if row["Expiry_Class"] == "Expired":
#             return "Expired"
#         elif row["Forecasted_Demand"] < row["Stock_Quantity"]:
#             return "High"
#         else:
#             return "Low"

#     df["Risk_Level"] = df.apply(assign_risk, axis=1)

#     df.to_csv(OUTPUT_PATH, index=False)
#     print(f"✅ Risk scoring complete. Results saved → {OUTPUT_PATH}")

#     print("\nRisk Level Distribution:")
#     print(df["Risk_Level"].value_counts())

# # Optional: allow standalone execution
# if __name__ == "__main__":
#     main()