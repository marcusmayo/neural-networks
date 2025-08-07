import requests
import json
import numpy as np

# Model server URL
url = "http://127.0.0.1:1234/invocations"

# Sample data that matches your model's input shape (53 features based on your successful curl test)
sample_features = [
    0.9948919228602967, -0.17211219589095383, -1.4668999385830706, 
    1.5736791319572767, 1.2873493468765707, -1.8516530591711033, 
    0.03189879581852941, -0.5830422962342866, 1.6927598598326044, 
    -0.5673610243790836, -0.5950058351843193, -0.29817742107022494, 
    -0.013139708280690915, -0.2891607112787806, -0.30431326550270593, 
    -0.3002320016231533, -0.3060031180395925, 3.3269730987791286, 
    -0.3060031180395925, -0.3045389363984451, -0.3002320016231533, 
    -0.3018235220673317, -0.3002320016231533, -0.30544051862468263, 
    -0.20849563376845656, -0.207139723612869, -0.20440538389169988, 
    -0.20440538389169988, -0.20744167607718714, -0.20864583663401953, 
    -0.2089459724643669, -0.20894597246436694, -0.20894597246436694, 
    -0.20894597246436694, -0.20894597246436694, -0.20909590582305057, 
    -0.20924574973887472, -0.20924574973887472, -0.20924574973887472, 
    -0.20939550440709528, -0.20939550440709528, -0.20909590582305057, 
    4.782494406400571, -0.20909590582305057, -0.20909590582305057, 
    -0.20909590582305057, -0.20909590582305057, -0.40789194515971317, 
    -0.40539375617774753, -0.40750803910152145, -0.40712397733604583, 
    -0.40865929257184186, 2.4327744901131347
]

# Test data - ensure all values are native Python types (not numpy types)
data = {
    "inputs": [
        sample_features,  # First sample (same as your successful curl test)
        [float(x * 0.9) for x in sample_features]  # Second sample with slight variation
    ]
}

headers = {
    "Content-Type": "application/json"
}

try:
    print("Making prediction request...")
    print(f"Number of features: {len(sample_features)}")
    print(f"Number of samples: {len(data['inputs'])}")
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Prediction successful!")
        print(f"Predictions: {result['predictions']}")
        
        # Pretty print results
        for i, pred in enumerate(result['predictions']):
            print(f"Sample {i+1}: {pred[0]:.2f} bike rentals predicted")
    else:
        print(f"\n❌ Error: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the model server.")
    print("Make sure the server is running: mlflow models serve -m 'models:/BikeRentalModel/3' --host 0.0.0.0 -p 1234")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON serialization error: {e}")
    print("This usually means there are numpy types or ellipsis in the data")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Test health endpoint
print("\n" + "="*50)
print("Testing health endpoint...")
try:
    health_response = requests.get("http://127.0.0.1:1234/health")
    print(f"✅ Health check: {health_response.status_code}")
except Exception as e:
    print(f"❌ Health check failed: {e}")

# Test version endpoint
print("\nTesting version endpoint...")
try:
    version_response = requests.get("http://127.0.0.1:1234/version")
    if version_response.status_code == 200:
        version_info = version_response.json()
        print(f"✅ Model version info: {version_info}")
    else:
        print(f"❌ Version check failed: {version_response.status_code}")
except Exception as e:
    print(f"❌ Version check failed: {e}")
