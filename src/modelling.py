# src/modelling.py
"""
Model Training & Evaluation (Updated with Label Encoding)
---------------------------------------------------------
- Uses hypothesis-driven significant features
- Expands to include Days_Until_Expiry, Remaining_Shelf_Life_Ratio, Stock_Age
- Adds GridSearchCV tuning for RF, GB, XGB
- Uses SMOTE for class balancing
- Handles label encoding for Expiry_Class (needed for XGBoost)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# -------------------------
# Load processed data
# -------------------------
df = pd.read_csv(
    "/Users/sakshizanjad/Desktop/grocery_expiry_project/data/processed/processed_data.csv"
)

# -------------------------
# Feature selection (expanded safe set)
# -------------------------
selected_features = [
    "Category",
    "Shelf_Life",
    "Stock_Quantity",
    "Stock_Value",
    "Sales_Volume",
    "Inventory_Turnover_Rate",
    "Unit_Price",
    "Days_Until_Expiry",
    "Remaining_Shelf_Life_Ratio",
    "Stock_Age"
]

print(f"\n🚀 Running with EXPANDED SAFE features: {selected_features}\n")

X = df[selected_features]
y = df["Expiry_Class"]

# -------------------------
# Encode target labels
# -------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)   # XGBoost needs numeric labels

# Save encoder for later use (decoding predictions)
joblib.dump(le, "models/label_encoder.pkl")

# -------------------------
# One-hot encode categorical features
# -------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# -------------------------
# Define models + grids
# -------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        },
        cv=3, scoring="f1_macro", n_jobs=-1
    ),
    "Gradient Boosting": GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        cv=3, scoring="f1_macro", n_jobs=-1
    ),
    "XGBoost": GridSearchCV(
        xgb.XGBClassifier(
            random_state=42,
            eval_metric="mlogloss"
        ),
        param_grid={
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, 7]
        },
        cv=3, scoring="f1_macro", n_jobs=-1
    )
}

# -------------------------
# Train & evaluate
# -------------------------
results = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    if name == "Logistic Regression":
        # Logistic Regression handles imbalance with class_weight
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        # Oversample with SMOTE for tree models
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    try:
        roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo", average="macro")
    except Exception:
        roc = None

    print(f"Accuracy: {acc}")
    print(f"F1 (macro): {f1}")
    if roc:
        print(f"ROC AUC (macro): {roc}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

    results[name] = {"accuracy": acc, "f1_macro": f1, "roc_auc": roc}

# -------------------------
# Save best model
# -------------------------
print("\n✅ Done. Results summary:", results)
best_model = max(results, key=lambda x: results[x]["f1_macro"])
print(f"\n🏆 Best model: {best_model} with F1 = {results[best_model]['f1_macro']}")

final_model = models[best_model]
if isinstance(final_model, GridSearchCV):
    final_model = final_model.best_estimator_

joblib.dump(final_model, "models/best_model.pkl")
print("✅ Best model + label encoder saved to /models/")