"""
Quick check to see current category status and fix if needed
"""
import pandas as pd
import os
import sys

def check_forecast_files():
    """Check existing forecast files for category issues"""
    print("=== CHECKING FORECAST FILES ===")
    
    if not os.path.exists('forecasts'):
        print("No forecasts directory found")
        return
    
    forecast_files = [f for f in os.listdir('forecasts') if f.endswith('.xlsx')]
    if not forecast_files:
        print("No Excel files in forecasts directory")
        return
    
    print(f"Found {len(forecast_files)} forecast files:")
    for file in forecast_files:
        print(f"  - {file}")
    
    # Check the latest file
    latest_file = max(forecast_files, key=lambda x: os.path.getctime(f'forecasts/{x}'))
    print(f"\nAnalyzing latest file: {latest_file}")
    
    try:
        df = pd.read_excel(f'forecasts/{latest_file}')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if 'Category' in df.columns:
            print("\n📊 CATEGORY ANALYSIS:")
            category_counts = df['Category'].value_counts()
            print("Category distribution:")
            for cat, count in category_counts.items():
                print(f"  {cat}: {count}")
            
            # Check if all are "Unknown"
            if len(category_counts) == 1 and 'Unknown' in category_counts:
                print("🚨 PROBLEM FOUND: All categories are 'Unknown'")
                return False
            else:
                print("✅ Categories look good!")
                return True
        else:
            print("❌ No 'Category' column found")
            return False
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def check_api_logic():
    """Check the API category extraction logic"""
    print("\n=== CHECKING API LOGIC ===")
    
    try:
        # Read the optimized_api.py file to check exact_matches
        with open('optimized_api.py', 'r') as f:
            content = f.read()
            
        if "exact_matches = ['AIMS Tkt Category'" in content:
            print("✅ 'AIMS Tkt Category' is in exact_matches list")
        else:
            print("❌ 'AIMS Tkt Category' not found in exact_matches")
            
        if "TktCategory" in content:
            print("✅ Found references to TktCategory in API")
        else:
            print("❌ No TktCategory references found")
            
    except Exception as e:
        print(f"Error checking API logic: {e}")

def suggest_fix():
    """Suggest fix for the category issue"""
    print("\n=== SUGGESTED FIX ===")
    print("""
The issue is likely that the data preprocessing creates 'AIMS Tkt Category' 
but the prediction/export logic isn't finding it properly.

The fix is to ensure the category extraction in optimized_api.py 
has robust column matching. The current exact_matches list should include:
- 'AIMS Tkt Category' (from data preprocessing)
- 'Ticket Category' 
- Other variations

If categories are still showing as 'Unknown', it means the category extraction
is failing in the generate_predictions function around line 488-525.
    """)

if __name__ == "__main__":
    print("🔍 AIMS CATEGORY STATUS CHECK\n")
    
    forecast_ok = check_forecast_files()
    check_api_logic()
    
    if not forecast_ok:
        suggest_fix()
    else:
        print("\n🎉 Categories appear to be working correctly!")
