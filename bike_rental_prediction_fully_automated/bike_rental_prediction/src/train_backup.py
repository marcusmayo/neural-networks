import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from preprocess import load_and_preprocess

# Detect environment
IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
IS_LOCAL = not IS_CI

# Set up safe paths BEFORE importing MLflow
def setup_safe_paths():
    """Configure all paths to use local directories only"""
    # Get absolute path to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create safe directories
    mlruns_dir = os.path.join(project_root, 'mlruns')
    temp_dir = os.path.join(mlruns_dir, 'temp')
    models_dir = os.path.join(project_root, 'models')
    
    # Create directories
    for directory in [mlruns_dir, temp_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Override ALL possible paths that MLflow might use
    os.environ['MLFLOW_TRACKING_URI'] = f'file://{mlruns_dir}'
    os.environ['MLFLOW_ARTIFACT_ROOT'] = mlruns_dir
    os.environ['MLFLOW_REGISTRY_URI'] = f'file://{mlruns_dir}'
    os.environ['TMPDIR'] = temp_dir
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir
    os.environ['HOME'] = project_root  # Override HOME to prevent /home/ubuntu access
    
    return mlruns_dir, temp_dir, models_dir

# Setup paths BEFORE any MLflow imports
MLRUNS_DIR, TEMP_DIR, MODELS_DIR = setup_safe_paths()

# Now safe to import MLflow
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.models.signature import infer_signature
    
    # Configure MLflow to use local paths
    mlflow.set_tracking_uri(f'file://{MLRUNS_DIR}')
    mlflow.set_registry_uri(f'file://{MLRUNS_DIR}')
    
    MLFLOW_AVAILABLE = True
    print(f"✅ MLflow configured with local tracking: {MLRUNS_DIR}")
    
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ℹ️ MLflow not available, will save model locally")

def safe_mlflow_log(func):
    """Decorator to safely handle MLflow operations"""
    def wrapper(*args, **kwargs):
        try:
            # Create a temporary directory for this specific run
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as run_temp:
                # Override temp directory for this operation
                old_tmpdir = os.environ.get('TMPDIR')
                os.environ['TMPDIR'] = run_temp
                
                result = func(*args, **kwargs)
                
                # Restore old tmpdir
                if old_tmpdir:
                    os.environ['TMPDIR'] = old_tmpdir
                    
                return result
        except Exception as e:
            print(f"⚠️ MLflow operation failed: {e}")
            print("Falling back to local save...")
            return None
    return wrapper

def train_model():
    print("🚀 Starting training...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 MLflow tracking: {os.environ.get('MLFLOW_TRACKING_URI', 'Not set')}")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)
    
    # Training loop
    epochs = 5 if IS_CI else 100
    print(f"🏃 Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_tensor)
                val_loss = loss_fn(val_output, y_test_tensor)
                print(f"  Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    print(f"✅ Training complete! Final loss: {loss.item():.4f}")
    
    # Save model
    model.eval()
    
    if MLFLOW_AVAILABLE and IS_LOCAL:
        save_with_mlflow(model, X_train, X_test, y_test, loss.item(), epochs)
    else:
        save_locally(model)
    
    return True

@safe_mlflow_log
def save_with_mlflow(model, X_train, X_test, y_test, final_loss, epochs):
    """Save model using MLflow with all safety measures"""
    print("📦 Attempting to log with MLflow...")
    
    try:
        # Create experiment with local artifact location
        experiment_name = "bike_rental_experiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f'file://{MLRUNS_DIR}/artifacts'
            )
        else:
            experiment_id = experiment.experiment_id
        
        # Start run with explicit artifact location
        with mlflow.start_run(experiment_id=experiment_id) as run:
            print(f"📊 MLflow run started: {run.info.run_id}")
            
            # Log parameters and metrics
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_metric("final_loss", final_loss)
            
            # Prepare input example and signature
            input_example = X_train[:5]
            with torch.no_grad():
                output_example = model(torch.tensor(input_example)).numpy()
            
            signature = infer_signature(X_train, output_example)
            
            # Create a temporary directory for the model
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as model_temp:
                model_path = os.path.join(model_temp, "model")
                
                # Log model with explicit local path
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature,
                    pip_requirements=["torch", "numpy", "scikit-learn"]
                )
                
                print(f"✅ Model logged to MLflow run: {run.info.run_id}")
                
                # Try to register model (may fail in CI)
                try:
                    model_uri = f"runs:/{run.info.run_id}/model"
                    mlflow.register_model(model_uri=model_uri, name="BikeRentalModel")
                    print(f"✅ Model registered: BikeRentalModel")
                except Exception as e:
                    print(f"ℹ️ Model registration skipped: {e}")
            
            return True
            
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
        print("Falling back to local save...")
        save_locally(model)
        return False

def save_locally(model):
    """Fallback method to save model locally"""
    print("💾 Saving model locally...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "model.pt")
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model)
    }, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    
    # Also save as scripted model for easier loading
    scripted_path = os.path.join(MODELS_DIR, "model_scripted.pt")
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, scripted_path)
    print(f"✅ Scripted model saved to: {scripted_path}")

def cleanup_temp():
    """Clean up temporary directories"""
    if os.path.exists(TEMP_DIR):
        try:
            # Remove only files, keep the directory
            for item in os.listdir(TEMP_DIR):
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"ℹ️ Cleanup note: {e}")

if __name__ == "__main__":
    try:
        success = train_model()
        cleanup_temp()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
