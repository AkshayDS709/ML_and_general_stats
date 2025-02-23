import streamlit as st
import pandas as pd
import joblib
import os

# Load data
df = pd.read_csv("data/demand_features.csv")
df['date'] = pd.to_datetime(df['date'])

# Load model directory
model_dir = "src/models/sku_models/"

st.title("Demand Forecasting Dashboard")

# Select SKU
sku_list = df['sku'].unique()
selected_sku = st.selectbox("Select a SKU:", sku_list)

# Filter data
sku_data = df[df['sku'] == selected_sku]

# Display historical demand
st.subheader("Historical Demand Trends")
st.line_chart(sku_data[['date', 'demand']].set_index('date'))

# Load the trained AutoARIMA model
model_path = os.path.join(model_dir, f"model_{selected_sku}.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)

    # Predict next week's demand
    predicted_demand = model.predict(n_periods=1)[0]

    st.subheader("Next Week's Demand Forecast")
    st.write(f"Predicted demand for SKU {selected_sku}: {predicted_demand:.2f} units")

    # Alert for replenishment order
    lead_time = st.slider("Supplier lead time (weeks):", 1, 8, 4)
    reorder_threshold = st.number_input("Reorder threshold:", min_value=0, value=50)

    if predicted_demand > reorder_threshold:
        st.warning(f"🚨 Replenishment alert: Consider ordering {int(predicted_demand)} units for SKU {selected_sku} (Lead time: {lead_time} weeks).")
    else:
        st.success(f"✅ No immediate replenishment needed for SKU {selected_sku}.")
else:
    st.error("Model not found for this SKU. Please train the model first.")
