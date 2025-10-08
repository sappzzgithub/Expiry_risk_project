# 📊 Grocery Expiry Risk Prediction Project

## Overview
This project presents an **Inventory Insights Dashboard** designed to combat grocery waste by predicting expiry risks, providing actionable recommendations, and visualizing key inventory trends. It integrates a robust backend pipeline—responsible for data preprocessing, demand forecasting, risk scoring, and recommendation generation—with an intuitive **Streamlit dashboard** for visualization and interaction.

---

## Key Features

### Backend Pipeline
The core engine for data processing and predictive analytics:
- **Data Preprocessing**: Cleans, validates, and transforms raw inventory and sales data.
- **Forecasting**: Predicts **future demand** and optimal **stock levels** to anticipate overstocking.
- **Risk Scoring**: Calculates and assigns **risk levels** (e.g., High, Medium, Low) for individual inventory items based on stock vs. predicted demand and expiry dates.
- **Recommendations**: Generates actionable inventory insights such as suggested **discounts**, **relocations** (e.g., front of store), or product **bundling** to mitigate identified risks.

### Streamlit Dashboard
An intuitive interface for monitoring and decision-making:
- **Key Metrics**: Displays at-a-glance figures like **Total Products**, **Expired Items**, **Near-Expiry Items**, and the overall **Inventory Risk Percentage**.
- **Visualizations**: Utilizes pie charts, bar charts, and line charts to visualize risk distributions, stock trends, and forecast vs. actual performance. Plots are interactive and **filterable** for deeper analysis.
- **Recommendations Interface**: Presents the suggested actions, offering a feature to **download recommendations** as a CSV file for implementation.
- **Interactive Filters**: Allows users to filter the entire dashboard by criteria like **risk levels** and **predicted actions**.

---

## Setup and Installation

### Prerequisites
Ensure you have the following installed on your system:
- **Python 3.8** or higher.
- **pip** (Python package installer).

### Step 1: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/sappzzgithub/Expiry_risk_project](https://github.com/sappzzgithub/Expiry_risk_project)
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
streamlit run app.py  # Assuming your main dashboard file is named app.py
```

The application will open in your web browser, typically at `http://localhost:8501`.

```
```
