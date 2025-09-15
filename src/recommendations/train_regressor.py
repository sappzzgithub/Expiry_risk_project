"""
Discount Regressor
------------------
- Trains RandomForestRegressor to predict discount percentage
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_regressor(df: pd.DataFrame, feature_cols):
    discount_df = df[df["Predicted_Action"] == "Discount"]
    if discount_df.empty:
        df["Predicted_Discount_Percent"] = 0
        return df

    Xd = discount_df[feature_cols]
    yd = discount_df["Discount_Percent"]

    Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42)
    reg.fit(Xd_train, yd_train)

    df.loc[df["Predicted_Action"] == "Discount", "Predicted_Discount_Percent"] = reg.predict(Xd)
    return df