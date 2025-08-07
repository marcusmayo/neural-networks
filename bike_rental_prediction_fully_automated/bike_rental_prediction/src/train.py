#!/usr/bin/env python3
import os
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from preprocess import load_and_preprocess

IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
print(f"🌍 Environment: {'GitHub Actions CI/CD' if IS_CI else 'Local Development'}")

class BikeRentalModel(nn.Module):
    def __init__(self, input_size):
        super(BikeRentalModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model():
    run_dir = Path("./runs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Saving to: {run_dir}")
    
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    model = BikeRentalModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).view(-1, 1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).view(-1, 1)
    
    epochs = 10 if IS_CI else 100
    print(f"🏃 Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_test_t), y_test_t)
                print(f"  Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'final_loss': loss.item()
    }, run_dir / "model.pt")
    
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump({
            'epochs': epochs,
            'final_train_loss': float(loss.item()),
            'final_val_loss': float(val_loss.item()),
            'timestamp': datetime.datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"✅ Training complete! Final loss: {loss.item():.4f}")
    return True

if __name__ == "__main__":
    try:
        train_model()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
