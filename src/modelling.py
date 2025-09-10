# src/modeling.py
"""
Expiry Risk Prediction using Mixed Features (Existing + Safe Engineered)
------------------------------------------------------------------------
This script trains multiple classifiers to predict Expiry_Class using:
- Existing business features (stock, sales, supplier info, etc.)
- Safe engineered features (Stock_Age, Stock_Value, Shelf_Life)

Leakage features (Days_Until_Expiry, Remaining_Shelf_Life_Ratio) are excluded.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# -------------------------
# Paths
# -------------------------
PROCESSED_PATH = "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/intermediate_data.csv"

# -------------------------
# Load Data
# -------------------------
date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
df = pd.read_csv(PROCESSED_PATH, parse_dates=date_cols)

# Target
y = df["Expiry_Class"].astype(str).str.strip()
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# -------------------------
# Mixed Feature Set (no leakage)
# -------------------------
mixed_features = [
    "Category", "Supplier_Name", "Warehouse_Location",
    "Stock_Quantity", "Reorder_Level", "Reorder_Quantity",
    "Unit_Price", "Sales_Volume", "Inventory_Turnover_Rate",
    "Stock_Age", "Stock_Value", "Shelf_Life"
]

X = df[mixed_features].copy()

# -------------------------
# Preprocessing
# -------------------------
categorical_cols = [c for c in ["Category", "Supplier_Name", "Warehouse_Location"] if c in X.columns]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# -------------------------
# Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
}

# -------------------------
# Train + Evaluate
# -------------------------
results = {}
print(f"\n🚀 Running with MIXED features: {mixed_features}\n")

for name, model in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # ROC AUC (macro-average for multiclass)
    try:
        y_proba = pipe.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    results[name] = {"accuracy": acc, "roc_auc": auc}

    print(f"\n🔹 {name} Results")
    print("Accuracy:", acc)
    if auc is not None:
        print("ROC AUC (macro):", auc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

print("\n✅ Done. Results summary:", results)

import joblib

# After training best model (say XGBoost pipeline)
best_model = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42))])
best_model.fit(X_train, y_train)

# Save model
joblib.dump(best_model, "/Users/sakshizanjad/Desktop/grocery_expiry_project/models/expiry_model.pkl")
print("✅ Model saved at models/expiry_model.pkl")

