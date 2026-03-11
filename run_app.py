import uvicorn
import logging

def run_optimized_api():
    """Run the Optimized AIMS API"""
    print("🎆 AIMS Incident Prediction System")
    print("=" * 60)
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔄 Alternative docs:   http://localhost:8000/redoc")
    print("❤️  Health check:      http://localhost:8000/status")
    print("📚 Examples & Guide:   http://localhost:8000/examples")
    print("=" * 60)
    print("🚀 Quick Start:")
    print("  1. Go to http://localhost:8000/docs")
    print("  2. Try POST /upload with your Excel file")
    print("  3. Use POST /train (xgboost, 3 months recommended)")
    print("  4. Generate predictions with POST /predict")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "optimized_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 API stopped gracefully.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTry: python optimized_api.py")

if __name__ == "__main__":
    run_optimized_api()