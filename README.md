# 📊 Product Expiry Risk Prediction Project

## Overview
This project presents an **Inventory Insights Dashboard** designed to combat grocery waste by predicting expiry risks, providing actionable recommendations, and visualizing key inventory trends. It integrates a robust backend pipeline—responsible for data preprocessing, demand forecasting, risk scoring, and recommendation generation—with an intuitive **Streamlit dashboard** for visualization and interaction.

---

## Key Features

### Backend Pipeline
The core engine for data processing and predictive analytics:
- **Data Preprocessing**: Cleans, validates, and transforms raw inventory and sales data (`src/data_preprocessing.py`).
- **Forecasting**: Predicts **future demand** and optimal **stock levels** using time-series models, with results stored in `forecasts/`.
- **Risk Scoring**: Calculates and assigns **risk levels** (e.g., High, Medium, Low) based on stock, predicted demand, and expiry dates (`src/risk_scoring.py`).
- **Recommendations**: Generates actionable inventory insights such as suggested **discounts**, **relocations**, or product **bundling** to mitigate identified risks (`src/recommendations/`).

### Streamlit Dashboard
An intuitive interface for monitoring and decision-making:
- **Key Metrics**: Displays at-a-glance figures like **Total Products**, **Expired Items**, **Near-Expiry Items**, and the overall **Inventory Risk Percentage** (`dashboard/app.py`).
- **Visualizations**: Uses interactive charts (pie, bar, line) to visualize risk distributions and stock trends.
- **Recommendations Interface**: Presents suggested actions, offering a feature to **download recommendations** as a CSV file.
- **Interactive Filters**: Allows users to filter the entire dashboard by criteria like **risk levels** and **predicted actions**.

---

## Project Structure

```
Expiry_risk_project/
├── data/                       # Stores all raw, interim, and final datasets 
│   ├── raw/                    # Original uploaded data (e.g., uploaded_inventory.csv) 
│   ├── processed/              # Cleaned data ready for modeling 
│   └── external/               # Final outputs (e.g., recommendations.csv, risk_scoring.csv) 
├── forecasts/                  # Stores detailed product-level demand forecasts 
│   └── product_level/          # Individual CSVs for each product's forecast
├── models/                     # Trained models and necessary artifacts 
│   ├── best_model.pkl          # Final trained risk prediction model 
│   └── label_encoder.pkl       # Label encoder for model preprocessing 
├── notebooks/                  # Exploratory Data Analysis (EDA) and experimentation
│   ├── EDA.ipynb               # Initial exploratory data analysis notebook
│   └── AdvanceEDA.ipynb        # Advanced exploratory data analysis notebook 
├── src/                        # Source code for the backend pipeline
│   ├── data_preprocessing.py   # Logic for cleaning and transforming data 
│   ├── forecasting.py          # Logic for time-series demand prediction 
│   ├── risk_scoring.py         # Logic for calculating inventory risk
│   ├── modelling.py            # Logic for training the risk prediction model 
│   └── recommendations/        # Module for generating mitigation actions
│       ├── __pycache__/        # Python compiled bytecode files 
│       ├── bootstrap_labels.py # Logic for bootstrapping labels 
│       ├── features.py         # Logic for feature engineering 
│       ├── recommend.py        # Logic for generating recommendations 
│       ├── train_classifier.py # Logic for training classifier model 
│       └── train_regressor.py  # Logic for training regressor model 
├── dashboard/                  # Streamlit application files
│   └── app.py                  # Main dashboard application 
├── run_pipeline.py             # Script to run the entire data and prediction pipeline 
├── requirements.txt            # List of required Python dependencies
└── README.md                   # Project overview and setup instructions 

````

---

## Setup and Installation

### Prerequisites
Ensure you have the following installed on your system:
- **Python 3.8** or higher.

### Step 1: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/sappzzgithub/Expiry_risk_project
cd Expiry_risk_project
````

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies:

```bash
python -m venv .venv

# For Windows:
.venv\Scripts\activate

# For Mac/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

Install all required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

Execute the Streamlit application from the project root directory:

```bash
streamlit run dashboard/app.py 
```

The application will open in your web browser, typically at `http://localhost:8501`.
