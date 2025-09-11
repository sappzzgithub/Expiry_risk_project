# src/recommendations/generate_labels.py
"""
Generate Synthetic Labels for Recommendations
---------------------------------------------
Creates action labels (Dispose, Discount, Reorder, Redistribute) and numeric
targets for training recommendation models.
"""

import pandas as pd
import os

# Paths
RISK_SCORES_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/external/risk_scores.csv"
OUTPUT_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/external/labeled_recommendations.csv"

def generate_labels():
    df = pd.read_csv(RISK_SCORES_PATH)

    # Initialize columns
    df["Action_Label"] = "None"
    df["Discount_Percent"] = 0
    df["Reorder_Qty"] = 0
    df["Redistribute_Qty"] = 0

    for idx, row in df.iterrows():
        if row["Expiry_Class"] == "Expired":
            df.at[idx, "Action_Label"] = "Dispose"

        elif row["Risk_Level"] == "High":
            if row["Forecasted_Demand"] < row["Stock_Quantity"]:
                df.at[idx, "Action_Label"] = "Discount"
                df.at[idx, "Discount_Percent"] = min(50, (row["Stock_Quantity"] - row["Forecasted_Demand"]) * 2)

        elif row["Risk_Level"] == "Medium":
            df.at[idx, "Action_Label"] = "Redistribute"
            df.at[idx, "Redistribute_Qty"] = int(row["Stock_Quantity"] * 0.3)

        else:  # Low risk
            df.at[idx, "Action_Label"] = "Reorder"
            df.at[idx, "Reorder_Qty"] = int(row["Forecasted_Demand"] * 0.2)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Labeled recommendations saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_labels()
