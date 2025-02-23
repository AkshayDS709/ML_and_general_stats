# Demand Forecasting Application

## Project Overview
This project implements a **demand forecasting application** for an artificial jewelry accessories company. 
The company moved away from spreadsheets to **automate replenishment orders** for suppliers, reducing costs on:
- **Inventory maintenance** (less capital stuck in stock)
- **Surge demand forecasting** (fewer lost sales due to stockouts)

## Features
- Forecast **weekly demand** for 5000+ SKUs
- **Top 700 products** account for 80% of sales
- **Weekly alerts** to reorder stock based on supplier turnaround time
- **Streamlit frontend** for visualization and decision-making
- **AutoARIMA models trained per SKU**

## Project Structure
```
Demand_Forecasting_App/
│── data/                 # Raw and processed data
│── notebooks/            # Jupyter notebooks for EDA and experiments
│── src/                  # Source code for the project
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model training and evaluation scripts
│   │   ├── sku_models/   # AutoARIMA models for each SKU
│   ├── utils/            # Utility functions
│   ├── frontend/         # Streamlit app code
│── reports/              # Reports and results
│── requirements.txt      # Dependencies list
│── README.md             # Project documentation
```

## Setup Instructions

### 1. Install dependencies
Ensure you have Python installed, then install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run feature engineering
This script creates additional features from the demand data:
```bash
python src/features/create_features.py
```

### 3. Train models
This script trains separate AutoARIMA models for each SKU and saves them:
```bash
python src/models/train_model.py
```

### 4. Run the Streamlit app
Launch the dashboard for demand forecasting:
```bash
streamlit run src/frontend/app.py
```

## How it Works
1. **Feature Engineering:** Creates new features like rolling averages, demand volatility, and lag-based features.
2. **Model Training:** Trains an **AutoARIMA model per SKU** to forecast demand.
3. **Streamlit Dashboard:** Allows users to view historical demand and predictions, providing alerts for replenishment based on supplier lead times.


Developed for demand forecasting and inventory optimization.
