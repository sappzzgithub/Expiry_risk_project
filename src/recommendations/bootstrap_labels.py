"""
Bootstrap Labels
----------------
- Generates initial 'Action' labels from Risk_Level
- Generates bootstrap 'Discount_Percent' for Discount actions
"""

import pandas as pd

def bootstrap_action(row):
    if row["Risk_Level"] == "Expired":
        return "Dispose"
    elif row["Risk_Level"] == "High":
        return "Discount"
    elif row["Risk_Level"] == "Low":
        # Decide between Bundle, Relocate, Monitor
        if row["Inventory_Turnover_Rate"] < 10 and row["Stock_Age"] > 180:
            return "Bundle"
        elif row["Warehouse_Location"].startswith("5") or row["Warehouse_Location"].startswith("9"):
            # Example heuristic: relocate certain warehouse locations
            return "Relocate"
        else:
            return "Monitor"
    else:
        return "Monitor"

def bootstrap_discount(row):
    if row["Action"] == "Discount":
        urgency_factor = max(0, 30 - row["Days_Until_Expiry"]) / 30
        stock_factor = (row["Stock_Quantity"] - row["Forecasted_Demand"]) / max(1, row["Stock_Quantity"])
        base_discount = 5 + urgency_factor * 25 + stock_factor * 20
        return min(max(base_discount, 5), 50)  # clamp 5–50%
    return 0

def add_bootstrap_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["Action"] = df.apply(bootstrap_action, axis=1)
    df["Discount_Percent"] = df.apply(bootstrap_discount, axis=1)
    return df