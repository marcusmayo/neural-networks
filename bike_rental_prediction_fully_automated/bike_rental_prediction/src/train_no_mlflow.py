#!/usr/bin/env python3
"""
Training script that completely bypasses MLflow to avoid permission issues.
This script will work in any environment without any external dependencies except PyTorch.
"""

import os
import sys
import json
import pickle
import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from preprocess import load_and_preprocess

# Detect environment
IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
IS_RESTRICTED = os.getenv('USER') == 'runner' or not os.access('/tmp', os.W_OK)

print("=" * 60)
print("🚀 BIKE RENTAL PREDICTION MODEL TRAINING")
print("=" * 60)
print(f"📍 Environment: {'CI/CD' if IS_CI else 'Local'}")
print(f"🔒 Restricted: {'Yes' if IS_RESTRICTED else 'No'}")
print(f"📁 Working dir: {os.getcwd()}")
print("=" * 60)

class BikeRentalModel(nn.Module):
    """Neural network model for bike rental prediction"""
    def __init__(self, input_size):
        super(BikeRentalModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_run_directory():
    """Create a directory for this training run"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    # Use local directory structure
    base_dir = Path("./training_runs")
    run_dir = base_dir / run_name
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    
    print(f"📁 Created run directory: {run_dir}")
    return run_dir

def save_metrics(run_dir, metrics):
    """Save training metrics to JSON file"""
    metrics_file = run_dir / "metrics" / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to: {metrics_file}")

def save_model(run_dir, model, model_info):
    """Save model and its information"""
    models_dir = run_dir / "models"
    
    # Save PyTorch model
    model_path = models_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': model_info
    }, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save scripted model for production
    try:
        scripted_model = torch.jit.script(model)
        scripted_path = models_dir / "model_scripted.pt"
        torch.jit.save(scripted_model, scripted_path)
        print(f"📦 Scripted model saved to: {scripted_path}")
    except Exception as e:
        print(f"⚠️ Could not save scripted model: {e}")
    
    # Save model info as JSON
    info_path = models_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"📄 Model info saved to: {info_path}")

def train_model():
    """Main training function"""
    
    # Create run directory
    run_dir = create_run_directory()
    
    # Load data
    print("\n📊 Loading and preprocessing data...")
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
    model = BikeRentalModel(input_size=X_train.shape[1])
    print(f"\n🤖 Model architecture:\n{model}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)
    
    # Training parameters
    epochs = 10 if IS_CI else 200
    best_val_loss = float('inf')
    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print(f"\n🏃 Starting training for {epochs} epochs...")
    print("-" * 40)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_output = model(X_train_tensor)
        train_loss = criterion(train_output, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_output = model(X_test_tensor)
            val_loss = criterion(val_output, y_test_tensor)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(float(train_loss.item()))
        training_history['val_loss'].append(float(val_loss.item()))
        training_history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_loss.item():.6f}, "
                  f"Val Loss: {val_loss.item():.6f}, "
                  f"LR: {current_lr:.6f}")
    
    print("-" * 40)
    print(f"✅ Training completed!")
    print(f"🏆 Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train_tensor)
        final_test_pred = model(X_test_tensor)
        
        train_mse = criterion(final_train_pred, y_train_tensor).item()
        test_mse = criterion(final_test_pred, y_test_tensor).item()
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Calculate R²
        train_r2 = 1 - (np.sum((y_train - final_train_pred.numpy().flatten())**2) / 
                       np.sum((y_train - np.mean(y_train))**2))
        test_r2 = 1 - (np.sum((y_test - final_test_pred.numpy().flatten())**2) / 
                      np.sum((y_test - np.mean(y_test))**2))
    
    # Prepare metrics
    metrics = {
        'training_config': {
            'epochs': epochs,
            'best_epoch': best_epoch,
            'initial_lr': 0.001,
            'optimizer': 'Adam',
            'loss_function': 'MSE'
        },
        'final_metrics': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'best_val_loss': float(best_val_loss)
        },
        'training_history': training_history,
        'data_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        }
    }
    
    # Model info
    model_info = {
        'architecture': str(model),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input_size': X_train.shape[1],
        'output_size': 1,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    # Save everything
    print(f"\n💾 Saving model and metrics...")
    save_metrics(run_dir, metrics)
    save_model(run_dir, model, model_info)
    
    # Create summary file
    summary_file = run_dir / "training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BIKE RENTAL PREDICTION MODEL - TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed at: {datetime.datetime.now()}\n")
        f.write(f"Environment: {'CI/CD' if IS_CI else 'Local'}\n\n")
        f.write("FINAL METRICS:\n")
        f.write(f"  Train RMSE: {train_rmse:.4f}\n")
        f.write(f"  Test RMSE: {test_rmse:.4f}\n")
        f.write(f"  Train R²: {train_r2:.4f}\n")
        f.write(f"  Test R²: {test_r2:.4f}\n\n")
        f.write(f"Best model from epoch: {best_epoch}/{epochs}\n")
        f.write(f"Run directory: {run_dir}\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📈 TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Model successfully trained and saved!")
    print(f"📊 Final Test RMSE: {test_rmse:.4f}")
    print(f"📊 Final Test R²: {test_r2:.4f}")
    print(f"📁 All artifacts saved to: {run_dir}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = train_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
