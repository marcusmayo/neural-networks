import os
import torch
import numpy as np
import tempfile
import sys

# Environment detection
IS_CI = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
IS_LOCAL = not IS_CI

if IS_LOCAL:
    try:
        import mlflow
        import mlflow.pytorch
        from mlflow.models.signature import infer_signature
        MLFLOW_AVAILABLE = True
        print("📊 MLflow available - using full tracking")
    except ImportError:
        MLFLOW_AVAILABLE = False
        print("⚠️ MLflow not available - fallback to local save")
else:
    MLFLOW_AVAILABLE = False
    print("🔄 CI environment detected - MLflow disabled")

from preprocess import load_and_preprocess

def train_model():
    print("🚀 Starting model training...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)

    model.train()
    for epoch in range(5 if IS_CI else 100):
        optimizer.zero_grad()
        loss = criterion(model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()

    print(f"✅ Training complete. Final loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        sample = model(X_train_tensor[:1]).item()
        print(f"✅ Sample prediction: {sample:.2f}")

    if MLFLOW_AVAILABLE and IS_LOCAL:
        print("📦 Logging with MLflow...")
        mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

        input_example = X_train[:1]
        with torch.no_grad():
            output_example = model(torch.tensor(input_example)).detach().numpy()
        signature = infer_signature(X_train, output_example)

        with mlflow.start_run() as run:
            mlflow.log_param("lr", 0.01)
            mlflow.log_param("epochs", 100)
            mlflow.log_metric("final_loss", loss.item())

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )

            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="BikeRentalModel")
            print(f"✅ Registered model: {model_uri}")
    else:
        print("💾 Saving locally without MLflow...")
        path = tempfile.mkdtemp() if IS_CI else "models"
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        print(f"✅ Model saved at {path}")

    return True

if __name__ == "__main__":
    success = train_model()
    if IS_CI:
        sys.exit(0 if success else 1)
