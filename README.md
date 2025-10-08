# 📊 Grocery Expiry Risk Prediction Project


## Overview
This project is an **Inventory Insights Dashboard** that predicts expiry risks, provides actionable recommendations, and visualizes inventory trends. It combines backend pipelines for data preprocessing, forecasting, risk scoring, and recommendations with an intuitive Streamlit dashboard.

---

## Features
### Backend Pipeline:
- **Data Preprocessing**: Cleans and transforms raw inventory data.
- **Forecasting**: Predicts future demand and stock levels.
- **Risk Scoring**: Calculates risk levels for inventory items.
- **Recommendations**: Generates actionable insights like discounts, relocations, and bundling.

### Dashboard:
- **Key Metrics**: Displays total products, expired items, near-expiry items, and inventory risk percentage.
- **Visualizations**:
  - Pie charts, bar charts, and line charts for risk levels and stock trends.
  - Filterable plots for deeper insights.
- **Recommendations**: Provides downloadable CSV files with actionable suggestions.
- **Interactive Filters**: Allows filtering by risk levels and predicted actions.

---

## Create and activate a virtual enviroment :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Mac/Linux



### Prerequisites:
- Python 3.8 or higher
- Streamlit
- Required Python libraries (listed in `requirements.txt`)
- pip install -r requirements.txt (run this to build the dependencies required)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/sappzzgithub/Expiry_risk_project>
   cd Expiry_risk_project