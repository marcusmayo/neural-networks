import os
import torch
import numpy as np
import tempfile
import sys

# Environment detection
IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
IS_LOCAL = not IS_CI

# Only import MLflow in local environment
if IS_LOCAL:
    try:
        import mlflow
        import mlflow.pytorch
        from mlflow.models.signature import infer_signature
        MLFLOW_AVAILABLE = True
        print("📊 MLflow available - running with full tracking")
    except ImportError:
        MLFLOW_AVAILABLE = False
        print("⚠️ MLflow not available - running in simple mode")
else:
    MLFLOW_AVAILABLE = False
    print("🔄 CI environment detected - MLflow disabled")

from preprocess import load_and_preprocess

def train_model():
    """
    Train the bike rental prediction model.
    Environment-aware: uses MLflow in local/production, simple mode in CI.
    """
    print("🚀 Starting model training...")
    print(f"Environment: {'CI' if IS_CI else 'Local/Production'}")
    
    try:
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess()
        print(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_train.shape[1]} features")
        
        # Ensure consistent data types
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32) 
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # Create model
        print("🧠 Creating neural network...")
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        # Training setup
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        # Environment-specific training
        epochs = 5 if IS_CI else 100
        print(f"🏃‍♂️ Training for {epochs} epochs...")
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Progress logging
            if (IS_CI and epoch % 2 == 0) or (IS_LOCAL and epoch % 20 == 0):
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        print(f"✅ Training completed. Final loss: {loss.item():.4f}")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            sample_pred = model(X_train_tensor[:1])
            print(f"✅ Sample prediction: {sample_pred.item():.2f}")
        
        if MLFLOW_AVAILABLE and IS_LOCAL:
            # Full MLflow integration for local/production
            print("📊 Logging with MLflow...")
            
            # Setup MLflow tracking URI (use local mlruns directory)
            mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
            
            # Prepare MLflow artifacts
            input_example = X_train[:1]
            model.eval()
            with torch.no_grad():
                prediction_example = model(torch.tensor(input_example, dtype=torch.float32)).detach().numpy()
            
            signature = infer_signature(X_train, prediction_example)
            
            # MLflow run
            with mlflow.start_run() as run:
                # Log parameters and metrics
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("learning_rate", 0.01)
                mlflow.log_param("hidden_units", 64)
                mlflow.log_param("input_features", X_train.shape[1])
                mlflow.log_metric("final_loss", loss.item())
                
                # Log model
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature
                )
                
                # Register model
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri=model_uri, name="BikeRentalModel")
                
                print(f"✅ Model registered: {model_uri}")
        
        else:
            # Simple mode (CI or no MLflow)
            print("💾 Saving model locally...")
            model_dir = tempfile.mkdtemp() if IS_CI else "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved to: {model_path}")
        
        print("🎉 Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if IS_CI:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    if IS_CI:
        sys.exit(0 if success else 1)
