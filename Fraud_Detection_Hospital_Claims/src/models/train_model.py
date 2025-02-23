import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/fraud_features.csv")
X = df.drop(columns=["claim_id", "is_fraudulent"])
y = df["is_fraudulent"]

# Handle class imbalance
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "HistGradientBoost": HistGradientBoostingClassifier(random_state=42),
}

# Train and evaluate models
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1 Score: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
    
    joblib.dump(model, f"src/models/{name}_model.pkl")

# Neural Network Model
class FraudNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Convert data to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Train Neural Network
nn_model = FraudNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluate Neural Network
y_pred_nn = (nn_model(X_test_tensor).detach().numpy() > 0.5).astype(int)
f1_nn = f1_score(y_test, y_pred_nn)
print(f"Neural Network F1 Score: {f1_nn:.4f}")

if f1_nn > best_f1:
    best_model = nn_model

joblib.dump(best_model, "src/models/best_model.pkl")
