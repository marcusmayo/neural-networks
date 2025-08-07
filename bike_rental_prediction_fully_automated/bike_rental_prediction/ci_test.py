#!/usr/bin/env python3
"""
CI Test Suite for Bike Rental Prediction API
"""
import sys
import json
import requests
import time

def test_endpoints():
    """Test all API endpoints"""
    
    # Use localhost for CI or EC2 host for integration tests
    API_URL = "http://18.233.252.250"
    
    print("🚀 Starting comprehensive CI tests")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Root endpoint
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "message" in data and "Bike Rental Prediction API" in data["message"]:
                print("✅ Root endpoint passed")
            else:
                print(f"❌ Root endpoint returned unexpected data: {data}")
                all_tests_passed = False
        else:
            print(f"❌ Root endpoint failed with status: {response.status_code}")
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        all_tests_passed = False
    
    # Test 2: Health endpoint
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("✅ Health check passed")
            else:
                print(f"❌ Health check returned: {data}")
                all_tests_passed = False
        else:
            print(f"❌ Health endpoint failed with status: {response.status_code}")
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        all_tests_passed = False
    
    # Test 3: Predict endpoint
    print("\n🔍 Testing predict endpoint...")
    test_features = [0.1] * 53  # 53 features as expected
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": test_features},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data and "status" in data:
                if data["status"] == "success":
                    print(f"✅ Prediction API response: {data}")
                else:
                    print(f"⚠️ Prediction returned non-success status: {data}")
                    all_tests_passed = False
            else:
                print(f"❌ Prediction response missing required fields: {data}")
                all_tests_passed = False
        else:
            print(f"❌ Predict endpoint failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")
        all_tests_passed = False
    
    # Test 4: Invalid input test
    print("\n🔍 Testing invalid input handling...")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": [0.1, 0.2]},  # Wrong number of features
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 400:
            print("✅ Invalid input properly rejected")
        else:
            print(f"⚠️ Invalid input returned status {response.status_code} (expected 400)")
    except Exception as e:
        print(f"⚠️ Invalid input test error: {e}")
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ ALL CI TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(test_endpoints())
