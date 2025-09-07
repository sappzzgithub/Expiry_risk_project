import pandas as pd
import numpy as np
import os

# Paths
RAW_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/raw/Grocery_Inventory_and_Sales_Balanced.csv"
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/intermediate_data.csv"

# Load data
df = pd.read_csv(RAW_PATH)

# Fix Column Names
if "Catagory" in df.columns:
    df.rename(columns={'Catagory': 'Category'}, inplace=True)

# Handle Missing Values
# Fill Category missing with default
if "Category" in df.columns:
    df['Category'] = df['Category'].fillna('Fruits & Vegetables')

# Fill numeric NaNs with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical NaNs with mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Ensure Correct Datatypes
# Clean and convert Unit_Price to float
df['Unit_Price'] = df['Unit_Price'].replace(r'[\$,]', '', regex=True).astype(float)

# Convert date columns (as requested, without loop)
df['Date_Received'] = pd.to_datetime(df['Date_Received'], errors='coerce')
df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'], errors='coerce')
df['Expiration_Date'] = pd.to_datetime(df['Expiration_Date'], errors='coerce')

# Remove duplicates
df = df.drop_duplicates()

# Feature Engineering
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
    df['Expiry_Class'] = df['Expiry_Class'].astype(str).str.strip()
    df['Expiry_Class'] = pd.Categorical(
        df['Expiry_Class'],
        categories=['Expired', 'Near_Expiry', 'Not_Expired']
    )

# 7️⃣ Save processed data
df.to_csv(PROCESSED_PATH, index=False)
print(f"✅ Preprocessing complete. Processed data saved at {PROCESSED_PATH}")
