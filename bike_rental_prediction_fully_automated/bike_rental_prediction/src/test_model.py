
import torch
from model import BikeNet
from preprocess import load_and_preprocess
from sklearn.metrics import mean_squared_error

def test_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = torch.load("models/latest_model.pt", map_location=torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()

    mse = mean_squared_error(y_test, predictions)
    print(f"✅ Test MSE: {mse:.4f}")
    return mse

if __name__ == "__main__":
    test_model()
