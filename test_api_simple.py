"""
Simple API Test Script for AIMS System
Tests all endpoints with proper error handling
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test API health"""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Data uploaded: {data['data_uploaded']}")
            print(f"   Available models: {data['available_models']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Start the server with: python run_app.py")
        return False

def test_examples():
    """Get API examples"""
    print("📚 Getting API examples...")
    try:
        response = requests.get(f"{API_URL}/examples")
        if response.status_code == 200:
            data = response.json()
            print("✅ Examples retrieved successfully!")
            print("Expected Excel format:")
            for key, value in data['excel_format']['example_row'].items():
                print(f"   {key}: {value}")
            return True
        else:
            print(f"❌ Failed to get examples: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_without_data():
    """Test prediction endpoint without data (should fail gracefully)"""
    print("🔮 Testing prediction without data...")
    payload = {
        "model_type": "xgboost",
        "horizon": 3
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 400:
            print("✅ Correctly rejected prediction without data!")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_training_without_data():
    """Test training without data (should fail gracefully)"""
    print("🚀 Testing training without data...")
    payload = {
        "model_type": "xgboost", 
        "horizon": 3
    }
    
    try:
        response = requests.post(f"{API_URL}/train", json=payload)
        if response.status_code == 400:
            print("✅ Correctly rejected training without data!")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AIMS API Simple Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_examples, 
        test_prediction_without_data,
        test_training_without_data
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print()
        if test():
            passed += 1
        else:
            failed += 1
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed! API is working correctly.")
        print("📖 Next: Go to http://localhost:8000/docs to try with real data")
    else:
        print("⚠️  Some tests failed. Check the server logs.")
    
    print("\n🔗 Useful Links:")
    print(f"   📖 Documentation: {API_URL}/docs")
    print(f"   📊 Status: {API_URL}/status")
    print(f"   📚 Examples: {API_URL}/examples")

if __name__ == "__main__":
    main()
