#!/usr/bin/env python3
"""
Simple API server for bike rental predictions
"""
from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import json

app = Flask(__name__)

# Load model on startup
MODEL_PATH = "/app/models/latest_model.pt"
model = None

class BikeRentalModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        # Assuming input size of 53 based on your test logs
        model = BikeRentalModel(53)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features'], dtype=np.float32)
        
        with torch.no_grad():
            input_tensor = torch.tensor(features).reshape(1, -1)
            prediction = model(input_tensor).item()
        
        return jsonify({
            "prediction": float(prediction),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=1234, debug=False)
