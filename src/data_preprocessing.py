#src/data_preprocessing.py

import pandas as pd
import numpy as np
import os

def main(uploaded_file_path):
    PROCESSED_PATH = "data/processed/processed_data.csv"

    df = pd.read_csv(uploaded_file_path)

    # Debugging: Log column names and data at key steps
    print("Columns in raw data:", df.columns.tolist())
    print("Sample data:")
    print(df.head())

    if "Unit_Price" in df.columns:
        df["Unit_Price"] = (
            df["Unit_Price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )

    def robust_datetime_convert(series):
        series = series.astype(str).str.strip().str.replace(r"[/]", "-", regex=True)
        return pd.to_datetime(series, errors="coerce")

    # Ensure required columns exist in the raw data
    required_columns = ["Date_Received", "Expiration_Date", "Last_Order_Date"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in raw data: {', '.join(missing_columns)}")

    # Convert date columns robustly
    for col in required_columns:
        df[col] = robust_datetime_convert(df[col])

    df = df.drop_duplicates()

    today = pd.to_datetime("today").normalize()

    if "Expiration_Date" in df.columns:
        df["Days_Until_Expiry"] = (df["Expiration_Date"] - today).dt.days
    if "Date_Received" in df.columns:
        df["Stock_Age"] = (today - df["Date_Received"]).dt.days
    if "Stock_Quantity" in df.columns and "Unit_Price" in df.columns:
        df["Stock_Value"] = df["Stock_Quantity"] * df["Unit_Price"]
    if "Date_Received" in df.columns and "Expiration_Date" in df.columns:
        df["Shelf_Life"] = (df["Expiration_Date"] - df["Date_Received"]).dt.days
        df["Remaining_Shelf_Life_Ratio"] = (
            df["Days_Until_Expiry"] / df["Shelf_Life"].replace(0, np.nan)
        ).clip(0, 1)

    if "Expiry_Class" in df.columns:
        df["Expiry_Class"] = df["Expiry_Class"].astype(str).str.strip()
        df["Expiry_Class"] = pd.Categorical(
            df["Expiry_Class"],
            categories=["Expired", "Near_Expiry", "Not_Expired"]
        )

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"✅ Preprocessing complete. Processed data saved at {PROCESSED_PATH}")

# Optional: allow standalone execution
if __name__ == "__main__":
    main("data/raw/merged_inventory.csv")