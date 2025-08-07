import os
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from preprocess import load_and_preprocess

def train_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Ensure consistent float32 dtype
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Define model
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Train the model
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Prepare input example and signature
    input_example = X_train[:1]  # Already float32 from above
    
    # Set model to eval mode for inference
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_example, dtype=torch.float32)
        prediction_example = model(input_tensor).detach().numpy()
    
    signature = infer_signature(X_train, prediction_example)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log some metrics for tracking
        mlflow.log_metric("final_loss", loss.item())
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("epochs", 100)
        mlflow.log_param("hidden_size", 64)
        
        # Log the model using MLflow's PyTorch integration
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )
        
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="BikeRentalModel")
        
        print(f"Model registered with URI: {model_uri}")

if __name__ == "__main__":
    train_model()
