
import torch
from torch import nn
from torch.optim import Adam
from model import BikeNet
from preprocess import load_and_preprocess
import mlflow
import mlflow.pytorch
import os

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = BikeNet(X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    os.makedirs("models", exist_ok=True)
    torch.save(model, "models/latest_model.pt")

    mlflow.set_experiment("BikePrediction")
    with mlflow.start_run():
        mlflow.log_param("epochs", 100)
        mlflow.log_metric("final_loss", loss.item())
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train_model()
