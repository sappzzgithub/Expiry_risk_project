# app.py
import sys
import os
import base64
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import forecasting, risk_scoring
from src.recommendations.recommend import run_recommendation_pipeline
from src.data_preprocessing import main as data_preprocessing_main
from src.forecasting import main as forecast_main
from src.risk_scoring import main as risk_main


# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(
    page_title="Product expiry risk prediction & optimisation platform",
    layout="wide",
    page_icon="📊"
)


# ----------------------- BACKGROUND IMAGE -----------------------
def set_background_image(image_path):
    """Encodes and applies a background image with dark overlay."""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background_image(os.path.join(os.path.dirname(__file__), "bg_image", "background.png"))



# ----------------------- CUSTOM STYLING -----------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Poppins:wght@600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #eaeaea;
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.6);
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .fade-container {
        animation: fadeIn 0.7s ease-in-out;
    }

    .glass-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .nav-btn {
        background: linear-gradient(90deg, #007bff, #6610f2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-size: 15px;
        font-weight: 600;
        transition: 0.3s;
        box-shadow: 0 0 10px rgba(102,16,242,0.3);
        cursor: pointer;
        margin: 0.3rem;
    }
    .nav-btn:hover {
        background: linear-gradient(90deg, #6610f2, #007bff);
        box-shadow: 0 0 15px rgba(0,123,255,0.6);
    }
    .active-btn {
        background: linear-gradient(90deg, #20c997, #17a2b8);
        box-shadow: 0 0 15px rgba(23,162,184,0.6);
    }
    footer {
        text-align: center;
        color: #999;
        margin-top: 2rem;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------- HEADER -----------------------
st.markdown("""
    <h1 style="text-align:center;">📊 Product Expiry Risk Prediction & Optimisation Platform</h1>
    <p style="text-align:center; color:#cccccc;">
    Explore inventory health, risks, forecasts, and AI-driven recommendations.
    </p>
""", unsafe_allow_html=True)


# ----------------------- FILE UPLOAD -----------------------
uploaded_file = st.file_uploader("Upload your Inventory CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    uploaded_df = pd.read_csv(uploaded_file)
    os.makedirs("data/raw", exist_ok=True)
    uploaded_path = "data/raw/uploaded_inventory.csv"
    uploaded_df.to_csv(uploaded_path, index=False)

    # Run pipeline silently
    data_preprocessing_main(uploaded_path)
    forecast_main()
    risk_main()
    run_recommendation_pipeline()

    # Load post-processing data
    risk_df = pd.read_csv("data/external/risk_scores.csv")
    rec_file = "data/external/recommendations.csv"
    rec_df = pd.read_csv(rec_file) if os.path.exists(rec_file) else None

    # ----------------------- NAVIGATION -----------------------
    st.markdown("<h3 style='text-align:center;'>Choose a Section</h3>", unsafe_allow_html=True)
    nav_cols = st.columns(5)
    sections = ["Overview", "Recommendations", "Filtered Insights", "Key Metrics", "Forecast Trends"]

    if "active_section" not in st.session_state:
        st.session_state.active_section = "Overview"

    for i, sec in enumerate(sections):
        btn_style = "active-btn" if st.session_state.active_section == sec else "nav-btn"
        if nav_cols[i].button(sec, key=sec, use_container_width=True):
            st.session_state.active_section = sec

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------- MAIN CONTENT -----------------------
    with st.container():
        st.markdown("<div class='glass-container fade-container'>", unsafe_allow_html=True)

        # ---- 1️⃣ Overview ----
        if st.session_state.active_section == "Overview":
            st.subheader("⚖️ Risk Level Overview")
            st.bar_chart(risk_df["Risk_Level"].value_counts())

        # ---- 2️⃣ Recommendations ----
        elif st.session_state.active_section == "Recommendations" and rec_df is not None:
            st.subheader("🎯 AI-Based Product Recommendations")
            st.dataframe(rec_df.head(10))
            st.download_button(
                label="⬇️ Download Recommendations CSV",
                data=rec_df.to_csv(index=False).encode('utf-8'),
                file_name="recommendations.csv",
                mime="text/csv"
            )

            st.subheader("📦 Top 10 Products by Stock Quantity")
            top_products = rec_df.groupby("Product_Name")["Stock_Quantity"].sum().nlargest(10).reset_index()
            st.bar_chart(top_products.set_index("Product_Name"))

        # ---- 3️⃣ Filtered Insights ----
        elif st.session_state.active_section == "Filtered Insights" and rec_df is not None:
            st.sidebar.header("🔍 Filter Options")
            risk_levels = rec_df["Risk_Level"].unique()
            selected_risks = st.sidebar.multiselect("Select Risk Levels", risk_levels, default=risk_levels)
            actions = rec_df["Predicted_Action"].unique()
            selected_actions = st.sidebar.multiselect("Select Actions", actions, default=actions)

            filtered_df = rec_df[
                (rec_df["Risk_Level"].isin(selected_risks)) &
                (rec_df["Predicted_Action"].isin(selected_actions))
            ]

            st.subheader("📈 Risk Level Distribution")
            risk_counts = filtered_df["Risk_Level"].value_counts()
            st.plotly_chart(px.pie(
                names=risk_counts.index,
                values=risk_counts.values,
                title="Risk Level Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            ), use_container_width=True)

            st.subheader("🧠 Action Distribution by Risk Level")
            action_risk_dist = filtered_df.groupby(["Risk_Level", "Predicted_Action"]).size().unstack(fill_value=0)
            st.dataframe(action_risk_dist)

        # ---- 4️⃣ Key Metrics ----
        elif st.session_state.active_section == "Key Metrics":
            st.subheader("📊 Inventory Health Overview")
            if "Expiry_Class" in uploaded_df.columns:
                col1, col2, col3, col4 = st.columns(4)
                total = len(uploaded_df)
                expired = (uploaded_df["Expiry_Class"] == "Expired").sum()
                near_expiry = (uploaded_df["Expiry_Class"] == "Near_Expiry").sum()
                not_expired = (uploaded_df["Expiry_Class"] == "Not_Expired").sum()
                col1.metric("📦 Total Products", total)
                col2.metric("❌ Expired Items", expired)
                col3.metric("⚠️ Near-Expiry", near_expiry)
                col4.metric("✅ Not Expired", not_expired)
            else:
                st.warning("⚠️ 'Expiry_Class' column missing in uploaded data.")

        # ---- 5️⃣ Forecast Trends ----
        elif st.session_state.active_section == "Forecast Trends":
            st.subheader("📅 Forecasting & Trend Analysis")
            forecast_path = "forecasts/product_level/all_products_forecast.csv"
            try:
                forecast_df = pd.read_csv(forecast_path)
                forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Forecast'}, inplace=True)
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
                forecast_df = forecast_df.dropna(subset=['Date'])
                st.line_chart(forecast_df.set_index("Date")["Forecast"])
            except Exception as e:
                st.error(f"Error loading forecast data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------- FOOTER -----------------------
    st.markdown("<footer>Inventory Insights Dashboard | Version 3.1 | © 2025</footer>", unsafe_allow_html=True)

else:
    st.info("📁 Please upload a CSV file to begin analysis.")
