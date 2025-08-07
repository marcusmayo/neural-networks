#!/usr/bin/env python3
"""
Simple CI test script that validates the model training pipeline
without any MLflow dependencies or directory creation issues.
"""

import os
import sys
import torch
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_preprocessing():
    """Test data preprocessing works"""
    print("🧪 Testing data preprocessing...")
    
    try:
        from preprocess import load_and_preprocess
        X_train, X_test, y_train, y_test = load_and_preprocess()
        
        print(f"✅ Data shapes: Train({X_train.shape}), Test({X_test.shape})")
        print(f"✅ Target ranges: Train({y_train.min():.1f}-{y_train.max():.1f}), Test({y_test.min():.1f}-{y_test.max():.1f})")
        
        # Basic validation
        assert X_train.shape[0] > 0, "Training data is empty"
        assert X_train.shape[1] > 0, "No features in training data"
        assert len(y_train) == X_train.shape[0], "Mismatched X and y training data"
        assert len(y_test) == X_test.shape[0], "Mismatched X and y test data"
        
        print("✅ Data preprocessing test PASSED")
        return True, X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Data preprocessing test FAILED: {e}")
        return False, None, None, None, None

def test_model_creation_and_training(X_train, X_test, y_train, y_test):
    """Test model creation and basic training"""
    print("🧪 Testing model creation and training...")
    
    try:
        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Train for a few epochs
        model.train()
        initial_loss = None
        
        for epoch in range(5):  # Just 5 epochs for CI
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
            
            if epoch == 4:  # Last epoch
                final_loss = loss.item()
                print(f"✅ Training: Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor[:1])
            train_pred = model(X_train_tensor[:1])
            
            print(f"✅ Sample predictions: Train={train_pred.item():.2f}, Test={test_pred.item():.2f}")
        
        # Validate predictions are reasonable (bike rental counts should be positive)
        assert train_pred.item() >= 0, "Prediction should be non-negative"
        assert test_pred.item() >= 0, "Prediction should be non-negative"
        assert train_pred.item() < 10000, "Prediction seems too high"
        
        print("✅ Model training test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all CI tests"""
    print("🚀 Starting CI tests for bike rental prediction model")
    print("=" * 60)
    
    # Test 1: Data preprocessing
    success, X_train, X_test, y_train, y_test = test_preprocessing()
    if not success:
        print("❌ CI tests FAILED at preprocessing stage")
        return False
    
    # Test 2: Model training
    success = test_model_creation_and_training(X_train, X_test, y_train, y_test)
    if not success:
        print("❌ CI tests FAILED at model training stage")
        return False
    
    print("=" * 60)
    print("🎉 ALL CI TESTS PASSED!")
    print("✅ Data preprocessing works correctly")
    print("✅ Model can be created and trained")
    print("✅ Predictions are generated successfully")
    print("✅ Neural network architecture is valid")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
