import pandas as pd

def load_data(file_path):
    """Load CSV data into Pandas DataFrame."""
    return pd.read_csv(file_path, parse_dates=['date'])

def save_data(df, file_path):
    """Save DataFrame as CSV."""
    df.to_csv(file_path, index=False)
