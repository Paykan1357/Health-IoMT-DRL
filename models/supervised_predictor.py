
# ============================================================
# supervised_predictor.py
# LSTM-based predictor for health_event (0,1,2,3)
# ============================================================

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class HealthEventPredictor(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def train_predictor(train_loader, val_loader, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HealthEventPredictor(input_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # ارزیابی هر ۱۰ epoch
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X).argmax(dim=-1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Acc={acc:.4f}")
    
    return model


def prepare_data_for_predictor(health_data, split="train", window_length=5):
    """
    استخراج داده‌های آموزشی برای مدل نظارت‌شده.
    """
    # دریافت لیست بیماران برای split مورد نظر
    if split == "train":
        patient_ids = health_data.patient_ids
    elif split == "val":
        patient_ids = health_data.patient_ids  # در اینجا باید val_ids را داشته باشید
    else:
        patient_ids = health_data.patient_ids
    
    X_list, y_list = [], []
    
    for pid in patient_ids:
        data = health_data.patient_data_cache[pid]["data"]
        # ۸ ویژگی اول (بدون health_event)
        features = data[:, :8]
        targets = data[:, 8].astype(np.int64)  # health_event
        
        for i in range(len(data) - window_length):
            X_list.append(features[i:i+window_length])
            y_list.append(targets[i+window_length-1])  # هدف آخرین گام
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def save_predictor(model, path="./models/predictor.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Predictor saved to {path}")


def load_predictor(path="./models/predictor.pth", device="cpu"):
    model = HealthEventPredictor(input_dim=8)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Predictor loaded from {path}")
    else:
        print(f"Predictor not found at {path}. Using untrained model.")
    model.to(device)
    model.eval()
    return model
