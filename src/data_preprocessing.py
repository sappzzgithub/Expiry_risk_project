import pandas as pd
import numpy as np
import os

# Paths
RAW_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/raw/Grocery_Inventory_and_Sales_Balanced.csv"
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/processed_data.csv"

# Load data
df = pd.read_csv(RAW_PATH)

# Fix Column Names
if "Catagory" in df.columns:
    df.rename(columns={'Catagory': 'Category'}, inplace=True)

# Handle Missing Values
if "Category" in df.columns:
    df['Category'].fillna('Fruits & Vegetables', inplace=True)

# Fill numeric NaNs with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NaNs with mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Ensure Correct Datatypes
if "Unit_Price" in df.columns:
    df['Unit_Price'] = df['Unit_Price'].replace('[\$,]', '', regex=True).astype(float)

# Convert date columns
date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Feature Engineering
today = pd.to_datetime("today")

if "Expiration_Date" in df.columns:
    df["Days_Until_Expiry"] = (df["Expiration_Date"] - today).dt.days

if "Date_Received" in df.columns:
    df["Stock_Age"] = (today - df["Date_Received"]).dt.days

if "Stock_Quantity" in df.columns and "Unit_Price" in df.columns:
    df["Stock_Value"] = df["Stock_Quantity"] * df["Unit_Price"]

# Shelf life & remaining ratio
if "Date_Received" in df.columns and "Expiration_Date" in df.columns:
    df["Shelf_Life"] = (df["Expiration_Date"] - df["Date_Received"]).dt.days
    df["Remaining_Shelf_Life_Ratio"] = (
        df["Days_Until_Expiry"] / df["Shelf_Life"].replace(0, np.nan)
    ).clip(0, 1)

# Ensure Expiry_Class is categorical for plotting
if "Expiry_Class" in df.columns:
    df['Expiry_Class'] = df['Expiry_Class'].str.strip()  # remove extra spaces
    df['Expiry_Class'] = pd.Categorical(
        df['Expiry_Class'],
        categories=['Expired', 'Near_Expiry', 'Not_Expired']
    )

# Save processed data
df.to_csv(PROCESSED_PATH, index=False)
print(f"✅ Preprocessing complete. Processed data saved at {PROCESSED_PATH}")