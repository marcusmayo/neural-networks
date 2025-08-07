#!/usr/bin/env python3
"""
Training script - NO MLflow, works everywhere!
"""
import os
import sys
import json
import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from preprocess import load_and_preprocess

# Detect environment
IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
print(f"Environment: {'GitHub Actions' if IS_CI else 'Local'}")
print(f"Working directory: {os.getcwd()}")

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
    """Train model without any MLflow"""
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("./runs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving outputs to: {run_dir}")
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Create model
    model = BikeRentalModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).view(-1, 1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).view(-1, 1)
    
    # Training
    epochs = 10 if IS_CI else 100
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validate
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_t)
                val_loss = criterion(val_output, y_test_t)
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")
    
    # Save model
    model_path = run_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[1]
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'epochs': epochs,
        'final_loss': float(loss.item()),
        'final_val_loss': float(val_loss.item())
    }
    
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete! Final loss: {loss.item():.4f}")
    return True

if __name__ == "__main__":
    try:
        train_model()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
