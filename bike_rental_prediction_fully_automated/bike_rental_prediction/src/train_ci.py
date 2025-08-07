import os
import torch
import numpy as np
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocess import load_and_preprocess

def train_model_ci():
    """
    CI-friendly training script that doesn't require MLflow registry access
    """
    print("🚀 Starting CI training test...")
    
    try:
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess()
        print(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_train.shape[1]} features")
        
        # Define model
        print("🧠 Creating neural network model...")
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        print(f"✅ Model created with input size: {X_train.shape[1]}")
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Convert data to tensors
        print("🔄 Converting data to tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Train the model (reduced epochs for CI)
        print("🏃‍♂️ Training model (10 epochs for CI)...")
        model.train()
        for epoch in range(10):  # Reduced epochs for CI
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Test the model
        print("🧪 Testing model predictions...")
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            print(f"✅ Test loss: {test_loss.item():.4f}")
            
            # Make a sample prediction
            sample_prediction = model(X_train_tensor[:1])
            print(f"✅ Sample prediction: {sample_prediction.item():.2f}")
        
        # Save model to temporary location for CI
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "test_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to: {model_path}")
        
        print("🎉 CI training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CI training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model_ci()
    sys.exit(0 if success else 1)
