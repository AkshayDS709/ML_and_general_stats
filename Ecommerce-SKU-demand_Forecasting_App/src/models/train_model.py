import pandas as pd
import joblib
from pmdarima import auto_arima
import os

# Load data
df = pd.read_csv("data/demand_features.csv")
df['date'] = pd.to_datetime(df['date'])

# Create model directory
model_dir = "src/models/sku_models/"
os.makedirs(model_dir, exist_ok=True)

# Train an AutoARIMA model for each SKU
unique_skus = df['sku'].unique()

for sku in unique_skus:
    print(f"Training model for SKU: {sku}")
    sku_data = df[df['sku'] == sku].set_index('date')
    
    # Train AutoARIMA
    model = auto_arima(sku_data['demand'], seasonal=True, m=52, suppress_warnings=True)
    
    # Save the model
    model_path = os.path.join(model_dir, f"model_{sku}.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")
