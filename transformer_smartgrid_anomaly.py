
# IoT Sensor Data Transformer for Anomaly Detection in Smart Grids
# Dataset: https://www.kaggle.com/datasets/jeanmidev/smart-meter-energy-consumption

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import shap
import random

# Fix random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Step 1: Data Preprocessing
# -----------------------------
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df.fillna(method="ffill", inplace=True)
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_length=24):
        self.series = series
        self.seq_length = seq_length

    def __len__(self):
        return len(self.series) - self.seq_length

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.seq_length]
        y = self.series[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Step 2: Define Transformer Model
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, src):
        src = src.unsqueeze(1)  # (batch, 1, seq_len)
        output = self.transformer(src)
        out = self.linear(output[:, -1, :])
        return out.squeeze()

# -----------------------------
# Step 3: Training Loop
# -----------------------------
def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        losses = []
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")

# -----------------------------
# Step 4: Evaluation
# -----------------------------
def detect_anomalies(y_true, y_pred, threshold=0.05):
    residual = np.abs(y_true - y_pred)
    anomaly = residual > threshold
    return anomaly.astype(int)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load dataset
    df = load_data("smart_meter.csv")
    values = df["energy_consumed"].values

    # Prepare dataset
    dataset = TimeSeriesDataset(values, seq_length=24)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model instantiation
    model = TransformerModel(input_dim=24)
    train_model(model, dataloader, epochs=5)

    # Prediction for anomaly detection
    test_dataset = TimeSeriesDataset(values[-500:], seq_length=24)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_preds, y_trues = [], []
    for X, y in test_loader:
        y_pred = model(X).item()
        y_preds.append(y_pred)
        y_trues.append(y.item())

    anomalies = detect_anomalies(np.array(y_trues), np.array(y_preds))
    print("Precision:", precision_score(y_trues, anomalies))
    print("Recall:", recall_score(y_trues, anomalies))

    # Visualization
    plt.figure(figsize=(15,5))
    plt.plot(y_trues, label="Actual")
    plt.plot(y_preds, label="Predicted")
    plt.scatter(np.arange(len(anomalies)), anomalies * max(y_trues), color='red', label="Anomalies", marker="x")
    plt.legend()
    plt.title("Transformer-Based Anomaly Detection in Smart Meter Data")
    plt.savefig("anomaly_detection_plot.png")
    plt.show()
