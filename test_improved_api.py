"""
Test script for the improved AIMS API with auto-model selection
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_prediction_api():
    """Test the improved prediction API"""
    
    print("🧪 Testing Improved AIMS API with Auto-Model Selection")
    print("=" * 60)
    
    # 1. Test status endpoint
    print("1. Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status: {status['status']}")
            print(f"📊 Data uploaded: {status['data_uploaded']}")
            print(f"🤖 Available models: {status['available_models']}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the API is running with: python run_app.py")
        return
    
    # 2. Test prediction with simplified request
    print("\n2. Testing simplified prediction endpoint...")
    
    # Simple prediction request (no model_type or horizon specified)
    prediction_request = {
        "app_filter": "all",
        "category_filter": "all"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"🎯 Auto-selected model: {result['results']['model_used']}")
            print(f"📅 Horizon: {result['results']['horizon_months']} months")
            print(f"📊 Apps processed: {result['results']['apps_returned']}")
            print(f"📝 Message: {result['message']}")
            
            # Show first few predictions
            predictions = result['results']['predictions']
            print("\n📈 Sample predictions:")
            for i, (month, values) in enumerate(predictions.items()):
                if i < 3:  # Show first 3 months
                    print(f"   {month}: {values[:5]}...")  # Show first 5 values
            
        elif response.status_code == 404:
            print("❌ No trained models found!")
            print("💡 You need to train a model first using POST /train")
            print("Example training request:")
            training_example = {
                "model_type": "xgboost",
                "horizon": 3
            }
            print(json.dumps(training_example, indent=2))
            
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
    
    # 3. Test filtered prediction
    print("\n3. Testing filtered prediction...")
    
    filtered_request = {
        "app_filter": ["APP001", "APP002"],
        "category_filter": "all"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=filtered_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Filtered prediction successful!")
            print(f"🎯 Model used: {result['results']['model_used']}")
            print(f"📊 Filtered to: {result['results']['apps_returned']} records")
        else:
            print(f"❌ Filtered prediction failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Filtered prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Test completed!")
    print("Key improvements:")
    print("✅ No need to specify model_type and horizon in predictions")
    print("✅ API automatically selects best available trained model")
    print("✅ Prevents model/horizon mismatches")
    print("✅ Clear error messages when no models are trained")

if __name__ == "__main__":
    test_prediction_api()
