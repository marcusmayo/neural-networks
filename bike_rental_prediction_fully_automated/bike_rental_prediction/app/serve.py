#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
import json

app = Flask(__name__)

class BikeRentalModel(nn.Module):
    def __init__(self, input_size=53):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Global model variable
model = None

def load_model():
    """Load the model - either from file or create new one"""
    global model
    model_path = "/app/models/latest_model.pt"
    
    model = BikeRentalModel(53)
    
    # Try to load weights if they exist
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If it's just the state dict directly
                model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized model")
    else:
        print(f"No model found at {model_path}, using randomly initialized model")
    
    model.eval()
    return model

# Load model on startup
model = load_model()

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "Bike Rental Prediction API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Support both 'features' and 'input' keys
        features = data.get('features') or data.get('input')
        
        if features is None:
            return jsonify({"error": "No features provided in request"}), 400
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Check feature dimensions
        if len(features) != 53:
            return jsonify({
                "error": f"Expected 53 features, got {len(features)}"
            }), 400
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.tensor(features).reshape(1, -1)
            prediction = model(input_tensor)
            prediction_value = prediction.item()
        
        return jsonify({
            "prediction": float(prediction_value),
            "status": "success"
        })
    
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    print("Starting Bike Rental Prediction API...")
    print(f"Model loaded: {model is not None}")
    app.run(host='0.0.0.0', port=1234, debug=False)
