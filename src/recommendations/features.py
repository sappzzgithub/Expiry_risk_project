"""
Feature Engineering
-------------------
- Selects relevant features
- Encodes categorical variables
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df: pd.DataFrame):
    feature_cols = [
        "Stock_Quantity", "Reorder_Level", "Reorder_Quantity", "Unit_Price",
        "Sales_Volume", "Inventory_Turnover_Rate", "Days_Until_Expiry",
        "Stock_Age", "Stock_Value", "Shelf_Life", "Remaining_Shelf_Life_Ratio",
        "Forecasted_Demand"
    ]

    # Encode Risk_Level
    risk_encoder = LabelEncoder()
    df["Risk_Level_Encoded"] = risk_encoder.fit_transform(df["Risk_Level"])
    feature_cols.append("Risk_Level_Encoded")

    X = df[feature_cols]
    return X, feature_cols