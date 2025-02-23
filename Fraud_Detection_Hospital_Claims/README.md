# Fraud Detection on Hospital Insurance Claims

## Project Overview
This project predicts potential **fraudulent physician/provider claims** using various machine learning models.

## Features
- **Multi-model approach**: XGBoost, Histogram Gradient Boosting, Random Forest, AdaBoost, and a Neural Network (PyTorch)
- **Automated model selection**: Picks the best model based on F1-score
- **Handles class imbalance** using SMOTE

## Project Structure
```
Fraud_Detection_Hospital_Claims/
│── data/                 # Raw and processed data
│── notebooks/            # Jupyter notebooks for EDA
│── src/                  # Source code
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model training and selection scripts
│   ├── utils/            # Utility functions
│── reports/              # Model performance reports
│── README.md             # Documentation
```

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run feature engineering:
   ```bash
   python src/features/create_features.py
   ```
3. Train models:
   ```bash
   python src/models/train_model.py
   ```

## License
Open-source project for research and educational purposes.

