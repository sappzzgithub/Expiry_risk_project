# Pseudocode inside recommendations.py

# Step 1: Load data
data = pd.read_csv("/Users/sakshizanjad/Desktop/grocery_expiry_project/data/external/risk_scores.csv")

# Step 2: Train/Test split for regression (discount %)
train_regressor(X, y_discount)

# Step 3: Train/Test split for classification (actions)
train_classifier(X, y_action)

# Step 4: Hybrid Recommendation Function
def generate_recommendation(row, reg_model, clf_model):
    if row["Risk_Level"] == "Expired":
        return {"Action": "Dispose", "Discount": 0}
    
    # Predict discount %
    discount = reg_model.predict(row[features])
    
    # Predict action
    action = clf_model.predict(row[features])
    
    return {"Action": action, "Discount": discount}
