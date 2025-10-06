# app.py
import sys
import os

# --- Add project root to Python path so 'src' can be imported ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib

from src import data_preprocessing, forecasting, risk_scoring
from src.recommendations.recommend import run_recommendation_pipeline

# --- Streamlit page configuration ---
st.set_page_config(page_title="Inventory Recommendation Dashboard", layout="wide")

st.title("📦 Inventory Recommendation System")
st.markdown("""
Upload your inventory CSV and get actionable recommendations including risk level, suggested actions, and discounts.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    uploaded_df = pd.read_csv(uploaded_file)
    st.dataframe(uploaded_df.head(10))

    # --- Step 1: Preprocessing ---
    st.subheader("🔧 Step 1: Data Preprocessing")
    st.info("Cleaning and transforming your inventory data...")
    preprocessed_path = "data/processed/processed_data.csv"
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    uploaded_df.to_csv("data/raw/uploaded_inventory.csv", index=False)  # Save temporarily
    data_preprocessing.main("data/raw/uploaded_inventory.csv")
    st.success("✅ Preprocessing complete!")

    # --- Step 2: Forecasting ---
    st.subheader("📈 Step 2: Forecasting")
    forecast_file = "forecasts/product_level/all_products_forecast.csv"
    os.makedirs("forecasts/product_level", exist_ok=True)
    if os.path.exists(forecast_file):
        st.info("Existing forecasts found. Using saved forecast.")
    else:
        st.info("Generating forecasts using Prophet...")
        forecasting.main()
    st.success("✅ Forecasting complete!")

    # --- Step 3: Risk Scoring ---
    st.subheader("⚖️ Step 3: Risk Scoring")
    os.makedirs("data/external", exist_ok=True)
    risk_scoring.main()
    st.success("✅ Risk scoring complete!")

    # Show Risk Distribution
    risk_df = pd.read_csv("data/external/risk_scores.csv")
    st.bar_chart(risk_df["Risk_Level"].value_counts())

    # --- Step 4: Recommendations ---
    st.subheader("🎯 Step 4: Generating Recommendations")
    run_recommendation_pipeline()
    st.success("✅ Recommendations generated!")

    # Show sample recommendations
    rec_file = "data/external/recommendations.csv"
    if os.path.exists(rec_file):
        rec_df = pd.read_csv(rec_file)
        st.dataframe(rec_df.head(10))
    else:
        st.warning("⚠️ Recommendations file not found.")

    # --- Step 5: Download ---
    st.subheader("💾 Download Recommendations")
    if os.path.exists(rec_file):
        st.download_button(
            label="Download CSV",
            data=rec_df.to_csv(index=False).encode('utf-8'),
            file_name="recommendations.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a file and generate recommendations to enable download.")
else:
    st.info("Please upload a CSV file to get started.")
