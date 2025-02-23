import pandas as pd

def create_features(df):
    """Generate additional features for fraud detection."""
    df['claim_ratio'] = df['total_reimbursed'] / (df['total_billed'] + 1e-9)
    df['avg_claim_amount'] = df['total_claim_amount'] / (df['num_procedures'] + 1e-9)
    df['high_cost_procedure'] = (df['total_claim_amount'] > 5000).astype(int)
    df['multiple_diagnoses'] = (df['num_diagnoses'] > 5).astype(int)
    df = pd.get_dummies(df, columns=['procedure_code', 'gender', 'state'], drop_first=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/fraud_data.csv")
    df = create_features(df)
    df.to_csv("data/fraud_features.csv", index=False)
    print("Feature creation completed.")
