
from fastapi import FastAPI
import torch
from src.model import BikeNet

app = FastAPI()
model = torch.load("models/latest_model.pt", map_location=torch.device('cpu'))

@app.post("/predict")
def predict(features: list):
    x = torch.tensor([features], dtype=torch.float32)
    prediction = model(x).detach().numpy().tolist()[0]
    return {"prediction": prediction}
