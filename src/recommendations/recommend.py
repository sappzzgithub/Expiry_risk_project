"""
Recommendation Engine (Orchestrator)
------------------------------------
- Loads risk_scores.csv
- Bootstraps labels (Dispose, Discount, Bundle, Relocate, Monitor)
- Trains classifier & regressor
- Saves final recommendations
"""

import os
import pandas as pd
from .bootstrap_labels import add_bootstrap_labels
from .features import prepare_features
from .train_classifier import train_classifier
from .train_regressor import train_regressor

# -------------------------
# Paths
# -------------------------
RISK_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/external/risk_scores.csv"
OUTPUT_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/external/recommendations.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(RISK_PATH, parse_dates=["Date_Received", "Last_Order_Date", "Expiration_Date"])

# -------------------------
# Bootstrap labels
# -------------------------
df = add_bootstrap_labels(df)

# -------------------------
# Prepare features
# -------------------------
X, feature_cols = prepare_features(df)
y_action = df["Action"]

# -------------------------
# Train classifier
# -------------------------
clf, le = train_classifier(X, y_action)
df["Predicted_Action"] = le.inverse_transform(clf.predict(X))

# -------------------------
# Train regressor (only if discount)
# -------------------------
df = train_regressor(df, feature_cols)

# -------------------------
# Save recommendations
# -------------------------
output_cols = [
    "Product_ID", "Product_Name", "Category", "Supplier_Name",
    "Stock_Quantity", "Risk_Level", "Predicted_Action", "Predicted_Discount_Percent"
]
df[output_cols].to_csv(OUTPUT_PATH, index=False)

print(f"✅ Recommendations complete. Results saved → {OUTPUT_PATH}")

# Summary
print("\nAction Distribution:")
print(df["Predicted_Action"].value_counts())