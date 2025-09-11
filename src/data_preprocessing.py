# src/data_preprocessing.py
"""
Data Preprocessing Script
-------------------------
This script preprocesses the grocery inventory dataset:
1. Cleans datatypes
2. Removes duplicates
3. Adds feature engineering columns
4. Saves processed data
"""

import pandas as pd
import numpy as np
import os

# -------------------------
# Paths
# -------------------------
RAW_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/intermediate.csv"
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/processed_data.csv"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(RAW_PATH)

# -------------------------
# Ensure Correct Datatypes
# -------------------------
# Clean and convert Unit_Price to float
if "Unit_Price" in df.columns:
    df["Unit_Price"] = (
        df["Unit_Price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

# Robust datetime conversion
def robust_datetime_convert(series):
    series = series.astype(str).str.strip().str.replace(r"[/]", "-", regex=True)
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

# Convert date columns individually
if "Date_Received" in df.columns:
    df["Date_Received"] = robust_datetime_convert(df["Date_Received"])

if "Last_Order_Date" in df.columns:
    df["Last_Order_Date"] = robust_datetime_convert(df["Last_Order_Date"])

if "Expiration_Date" in df.columns:
    df["Expiration_Date"] = robust_datetime_convert(df["Expiration_Date"])

# -------------------------
# Remove duplicates
# -------------------------
df = df.drop_duplicates()

# -------------------------
# Feature Engineering
# -------------------------
today = pd.to_datetime("today").normalize()

# Days until expiry
if "Expiration_Date" in df.columns:
    df["Days_Until_Expiry"] = (df["Expiration_Date"] - today).dt.days

# Stock age
if "Date_Received" in df.columns:
    df["Stock_Age"] = (today - df["Date_Received"]).dt.days

# Stock value
if "Stock_Quantity" in df.columns and "Unit_Price" in df.columns:
    df["Stock_Value"] = df["Stock_Quantity"] * df["Unit_Price"]

# Shelf life & remaining shelf life ratio
if "Date_Received" in df.columns and "Expiration_Date" in df.columns:
    df["Shelf_Life"] = (df["Expiration_Date"] - df["Date_Received"]).dt.days
    df["Remaining_Shelf_Life_Ratio"] = (
        df["Days_Until_Expiry"] / df["Shelf_Life"].replace(0, np.nan)
    ).clip(0, 1)

# Ensure Expiry_Class is categorical
if "Expiry_Class" in df.columns:
    df["Expiry_Class"] = df["Expiry_Class"].astype(str).str.strip()
    df["Expiry_Class"] = pd.Categorical(
        df["Expiry_Class"],
        categories=["Expired", "Near_Expiry", "Not_Expired"]
    )

# -------------------------
# Save processed data
# -------------------------
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print(f"✅ Preprocessing complete. Processed data saved at {PROCESSED_PATH}")