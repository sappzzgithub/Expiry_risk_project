# src/modelling.py

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

def load_trained_model():
    """
    Loads the pre-trained best model and label encoder.
    Returns:
        model: Trained ML model
        label_encoder: Trained label encoder for Expiry_Class
    """
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("✅ Loaded best_model.pkl and label_encoder.pkl")
        return model, label_encoder
    except FileNotFoundError:
        raise FileNotFoundError(
            "❌ Trained model or label encoder not found! "
            "Run training script to generate best_model.pkl and label_encoder.pkl."
        )

def predict_expiry_class(df: pd.DataFrame):
    """
    Uses the saved model to predict the Expiry_Class for new data.

    Args:
        df (pd.DataFrame): Preprocessed dataframe ready for prediction.

    Returns:
        pd.Series: Predicted Expiry_Class values (decoded).
    """
    model, label_encoder = load_trained_model()

    # ✅ The same features used during training
    selected_features = [
        "Category", "Shelf_Life", "Stock_Quantity", "Stock_Value",
        "Sales_Volume", "Inventory_Turnover_Rate", "Unit_Price",
        "Days_Until_Expiry", "Remaining_Shelf_Life_Ratio", "Stock_Age"
    ]

    # Filter expected columns
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns for prediction: {missing_cols}")

    X = df[selected_features]
    X = pd.get_dummies(X, drop_first=True)

    # Align with training features
    model_features = model.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    preds_enc = model.predict(X)
    preds = label_encoder.inverse_transform(preds_enc)

    return preds

if __name__ == "__main__":
    # If still used directly for debugging
    try:
        sample_df = pd.read_csv(
            "C:/Users/T8665/OneDrive - LTIMindtree/Desktop/Expiry_risk_project/data/processed/processed_data.csv"
        )
        preds = predict_expiry_class(sample_df)
        print("Sample Predictions:", preds[:10])
    except Exception as e:
        print("❌ Error:", e)



# # src/modelling.py

# def train_models():
#     import pandas as pd
#     import numpy as np
#     import joblib
#     from sklearn.model_selection import train_test_split, GridSearchCV
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     from sklearn.pipeline import Pipeline
#     from sklearn.metrics import (
#         accuracy_score, roc_auc_score, classification_report,
#         confusion_matrix, f1_score
#     )
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#     from imblearn.over_sampling import SMOTE
#     import xgboost as xgb

#     # Load processed data
#     df = pd.read_csv(
#         "C:/Users/T8665/OneDrive - LTIMindtree/Desktop/Expiry_risk_project/data/processed/processed_data.csv"
#     )

#     selected_features = [
#         "Category", "Shelf_Life", "Stock_Quantity", "Stock_Value",
#         "Sales_Volume", "Inventory_Turnover_Rate", "Unit_Price",
#         "Days_Until_Expiry", "Remaining_Shelf_Life_Ratio", "Stock_Age"
#     ]

#     print(f"\n🚀 Running with EXPANDED SAFE features: {selected_features}\n")

#     X = df[selected_features]
#     y = df["Expiry_Class"]

#     le = LabelEncoder()
#     y_enc = le.fit_transform(y)
#     joblib.dump(le, "models/label_encoder.pkl")

#     X = pd.get_dummies(X, drop_first=True)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
#     )

#     models = {
#         "Logistic Regression": Pipeline([
#             ("scaler", StandardScaler()),
#             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
#         ]),
#         "Random Forest": GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             param_grid={
#                 "n_estimators": [100, 200],
#                 "max_depth": [5, 10, None],
#                 "min_samples_split": [2, 5]
#             },
#             cv=3, scoring="f1_macro", n_jobs=-1
#         ),
#         "Gradient Boosting": GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             param_grid={
#                 "n_estimators": [100, 200],
#                 "learning_rate": [0.05, 0.1],
#                 "max_depth": [3, 5]
#             },
#             cv=3, scoring="f1_macro", n_jobs=-1
#         ),
#         "XGBoost": GridSearchCV(
#             xgb.XGBClassifier(random_state=42, eval_metric="mlogloss"),
#             param_grid={
#                 "n_estimators": [100, 200],
#                 "learning_rate": [0.05, 0.1],
#                 "max_depth": [3, 5, 7]
#             },
#             cv=3, scoring="f1_macro", n_jobs=-1
#         )
#     }

#     results = {}

#     for name, model in models.items():
#         print(f"\n🔹 Training {name}...")

#         if name == "Logistic Regression":
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#         else:
#             sm = SMOTE(random_state=42)
#             X_res, y_res = sm.fit_resample(X_train, y_train)
#             model.fit(X_res, y_res)
#             y_pred = model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average="macro")
#         try:
#             roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo", average="macro")
#         except Exception:
#             roc = None

#         print(f"Accuracy: {acc}")
#         print(f"F1 (macro): {f1}")
#         if roc:
#             print(f"ROC AUC (macro): {roc}")
#         print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#         print("Classification Report:\n", classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

#         results[name] = {"accuracy": acc, "f1_macro": f1, "roc_auc": roc}

#     print("\n✅ Done. Results summary:", results)
#     best_model = max(results, key=lambda x: results[x]["f1_macro"])
#     print(f"\n🏆 Best model: {best_model} with F1 = {results[best_model]['f1_macro']}")

#     final_model = models[best_model]
#     if isinstance(final_model, GridSearchCV):
#         final_model = final_model.best_estimator_

#     joblib.dump(final_model, "models/best_model.pkl")
#     print("✅ Best model + label encoder saved to /models/")

# # Optional: allow standalone execution
# if __name__ == "__main__":
#     train_models()