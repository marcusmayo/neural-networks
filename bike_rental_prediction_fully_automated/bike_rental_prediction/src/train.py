#!/usr/bin/env python3
"""
Training script without MLflow - works everywhere!
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
print(f"🌍 Environment: {'GitHub Actions CI/CD' if IS_CI else 'Local Development'}")
print(f"📁 Working directory: {os.getcwd()}")

class BikeRentalModel(nn.Module):
    def __init__(self, input_size):
        super(BikeRentalModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model():
    """Main training function - NO MLFLOW!"""
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("./runs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Saving outputs to: {run_dir}")
    
    # Load and preprocess data
    print("📊 Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Create model
    model = BikeRentalModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).view(-1, 1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).view(-1, 1)
    
    # Training parameters
    epochs = 10 if IS_CI else 100  # Fewer epochs in CI for speed
    print(f"🏃 Training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_test_t)
            val_loss = criterion(val_output, y_test_t)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train_t)
        final_test_pred = model(X_test_t)
        
        train_mse = criterion(final_train_pred, y_train_t).item()
        test_mse = criterion(final_test_pred, y_test_t).item()
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Final Train RMSE: {train_rmse:.4f}")
    print(f"📊 Final Test RMSE: {test_rmse:.4f}")
    
    # Save model (PyTorch format)
    model_path = run_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'input_size': X_train.shape[1],
        'epochs': epochs,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'environment': 'CI/CD' if IS_CI else 'Local',
        'epochs': epochs,
        'final_metrics': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse)
        },
        'data_shape': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1]
        }
    }
    
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to: {metrics_path}")
    
    # Also save to models directory for easy access
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Copy latest model
    latest_model_path = models_dir / "latest_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'input_size': X_train.shape[1]
    }, latest_model_path)
    print(f"💾 Latest model copied to: {latest_model_path}")
    
    print(f"\n🎉 All outputs saved to: {run_dir}")
    return True

if __name__ == "__main__":
    try:
        success = train_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
