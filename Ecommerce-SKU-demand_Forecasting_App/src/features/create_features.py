import pandas as pd
import numpy as np

def create_features(df):
    """Generate time-based and demand-based features."""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

    # Rolling demand features
    df['rolling_mean'] = df.groupby('sku')['demand'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['rolling_std'] = df.groupby('sku')['demand'].transform(lambda x: x.rolling(window=4, min_periods=1).std())
    
    # Lag features
    df['lag_1'] = df.groupby('sku')['demand'].shift(1)
    df['lag_2'] = df.groupby('sku')['demand'].shift(2)
    df['lag_3'] = df.groupby('sku')['demand'].shift(3)
    
    # Demand volatility indicator
    df['volatility'] = df['rolling_std'] / (df['rolling_mean'] + 1e-9)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/demand_data.csv", parse_dates=['date'])
    df = create_features(df)
    df.to_csv("data/demand_features.csv", index=False)
    print("Feature creation completed.")
