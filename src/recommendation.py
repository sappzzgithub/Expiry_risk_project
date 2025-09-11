# src/recommendation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report
import joblib
import os

# ==============================
# 1. Load Processed Data
# ==============================
df = pd.read_csv("data/processed/risk_scored_data.csv")

# Ensure required columns exist
required_cols = ["Stock_Quantity", "Forecasted_Demand", "Shelf_Life", "Expiry_Class"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"❌ Missing column: {col}")

# ==============================
# 2. Feature Engineering
# ==============================
df["Stock_to_Forecast_Ratio"] = df["Stock_Quantity"] / (df["Forecasted_Demand"] + 1)

# Targets
df["Discount_Perc"] = np.clip((df["Stock_to_Forecast_Ratio"] - 1) * 10, 0, 30)
df["Discount_Perc"] += np.random.uniform(-2, 2, size=len(df))
df["Discount_Perc"] = df["Discount_Perc"].clip(0, 30)

df["Relocate"] = np.where(df["Stock_Quantity"] > df["Forecasted_Demand"] * 1.5, 1, 0)
df["Dispose"] = np.where(df["Expiry_Class"] == "Expired", 1, 0)

# Features and targets
X = df[["Stock_Quantity", "Forecasted_Demand", "Shelf_Life", "Stock_to_Forecast_Ratio"]]
y_reg = df["Discount_Perc"]
y_clf = df[["Relocate", "Dispose"]]

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ==============================
# 4. Train Models
# ==============================
# Regression for Discount %
reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
reg_model.fit(X_train, y_reg_train)

# Classification for Relocate & Dispose
clf_models = {}
for target in ["Relocate", "Dispose"]:
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_clf_train[target])
    clf_models[target] = clf

# ==============================
# 5. Predictions & Evaluation
# ==============================
# Regression
y_reg_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)

# Classification
y_clf_pred = pd.DataFrame(index=X_test.index)
for target, clf in clf_models.items():
    y_clf_pred[target] = clf.predict(X_test)

# ==============================
# 6. Save Outputs
# ==============================
out_path = "data/processed/ai_recommendations_split.csv"
model_path_reg = "models/discount_regressor.pkl"
model_path_clf = "models/classifiers.pkl"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save predictions
output = pd.DataFrame({
    "Discount_Perc": y_reg_pred,
    "Relocate": y_clf_pred["Relocate"],
    "Dispose": y_clf_pred["Dispose"]
})
output.to_csv(out_path, index=False)

# Save models
joblib.dump(reg_model, model_path_reg)
joblib.dump(clf_models, model_path_clf)

# ==============================
# 7. Results
# ==============================
print("\nSample Predictions:")
print(output.head())

print(f"\n📊 Regression Metrics:")
print(f"   MSE (Discount %): {mse:.2f}")
print(f"   MAE (Discount %): {mae:.2f}")

print("\n📊 Classification Reports:")
for target in ["Relocate", "Dispose"]:
    print(f"\n--- {target} ---")
    print(classification_report(y_clf_test[target], y_clf_pred[target]))
    
print(f"\n✅ AI Recommendations saved at {out_path}")
print(f"✅ Models saved at {model_path_reg} & {model_path_clf}")







# # src/recommendation.py
# """
# AI-Based Multi-Output Recommendation System (No Direct Risk Mapping)
# -------------------------------------------------------------------
# Predicts Discount %, Relocation, and Disposal actions for products
# based on features like Stock_to_Forecast_Ratio, Days_Until_Expiry, Shelf_Life, and Expiry_Class.
# """

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, accuracy_score
# import joblib

# # -------------------------
# # Paths
# # -------------------------
# RISK_DATA_PATH = "data/processed/risk_scored_data.csv"
# OUTPUT_PATH = "data/processed/ai_multioutput_recommendations.csv"
# MODEL_PATH = "models/ai_multioutput_recommender.pkl"

# # -------------------------
# # Load Data
# # -------------------------
# df = pd.read_csv(RISK_DATA_PATH)

# # -------------------------
# # Feature Engineering
# # -------------------------
# df["Expiry_Class"] = df["Expiry_Class"].astype(str)

# # Numeric features
# num_features = ["Stock_to_Forecast_Ratio", "Days_Until_Expiry", "Shelf_Life"]

# # Categorical features
# cat_features = ["Expiry_Class"]
# encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# encoded_cat = encoder.fit_transform(df[cat_features])
# encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_features))
# df = pd.concat([df.reset_index(drop=True), encoded_cat_df], axis=1)

# X = df[num_features + list(encoded_cat_df.columns)]

# # -------------------------
# # Target Engineering (Synthetic Labels)
# # -------------------------
# # Instead of directly using Risk_Score, define targets via thresholds
# def generate_discount(row):
#     if row["Days_Until_Expiry"] < 30 or row["Stock_to_Forecast_Ratio"] > 1.2:
#         return 25  # High discount
#     elif row["Days_Until_Expiry"] < 60 or row["Stock_to_Forecast_Ratio"] > 1.0:
#         return 15  # Medium discount
#     elif row["Days_Until_Expiry"] < 90:
#         return 5   # Low discount
#     else:
#         return 0   # No discount

# def generate_relocate(row):
#     return 1 if row["Stock_to_Forecast_Ratio"] > 1.0 and row["Days_Until_Expiry"] < 60 else 0

# def generate_dispose(row):
#     return 1 if row["Days_Until_Expiry"] <= 0 else 0

# df["Discount_Perc"] = df.apply(generate_discount, axis=1)
# df["Relocate"] = df.apply(generate_relocate, axis=1)
# df["Dispose"] = df.apply(generate_dispose, axis=1)

# y = df[["Discount_Perc", "Relocate", "Dispose"]]

# # -------------------------
# # Train-Test Split
# # -------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # -------------------------
# # Model Training
# # -------------------------
# model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
# model.fit(X_train, y_train)

# # -------------------------
# # Predictions
# # -------------------------
# y_pred = model.predict(X_test)
# y_pred_df = pd.DataFrame(np.round(y_pred), columns=y_test.columns)

# # -------------------------
# # Evaluation
# # -------------------------
# print("\nSample Predictions:")
# print(y_pred_df.head())

# mse = mean_squared_error(y_test, y_pred)
# print(f"\nMean Squared Error (all targets): {mse:.2f}")

# accuracy_relocate = accuracy_score(y_test["Relocate"], y_pred_df["Relocate"])
# accuracy_dispose = accuracy_score(y_test["Dispose"], y_pred_df["Dispose"])
# print(f"Accuracy - Relocate: {accuracy_relocate:.2f}")
# print(f"Accuracy - Dispose: {accuracy_dispose:.2f}")

# # -------------------------
# # Save Recommendations
# # -------------------------
# df_recommend = df.copy()
# df_recommend[["Discount_Perc", "Relocate", "Dispose"]] = model.predict(X)
# df_recommend[["Discount_Perc", "Relocate", "Dispose"]] = df_recommend[["Discount_Perc", "Relocate", "Dispose"]].round()
# df_recommend.to_csv(OUTPUT_PATH, index=False)
# print(f"\n✅ AI Multi-Output Recommendations saved at {OUTPUT_PATH}")

# # -------------------------
# # Save Model
# # -------------------------
# joblib.dump(model, MODEL_PATH)
# print(f"✅ Model saved at {MODEL_PATH}")