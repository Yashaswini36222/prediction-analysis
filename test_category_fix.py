"""
Quick test to verify category extraction works correctly
"""
import pandas as pd
from data_preprocessing import clean_excel_data
import os

def test_category_extraction():
    # Find Excel files
    excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
    
    if not excel_files:
        print("No Excel files found. Please upload your AIMS data Excel file.")
        return
    
    # Use the first Excel file found
    file_path = excel_files[0]
    print(f"Testing with file: {file_path}")
    
    try:
        # Process data using the same function as the API
        processed_data, scalers = clean_excel_data(file_path)
        
        print(f"\nProcessed data shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")
        
        # Test category extraction logic
        data = processed_data.copy()
        num_rows = min(len(data), 10)  # Test with first 10 rows
        
        # Simulate the same logic from optimized_api.py
        category_column = None
        categories = ["Unknown" for _ in range(num_rows)]
        
        # Strategy 1: Try exact column name matches
        exact_matches = ['AIMS Tkt Category', 'Ticket Category', 'Category', 'Type', 'Incident Type', 'Ticket Type', 'Request Type', 'TktCategory']
        for col in exact_matches:
            if col in data.columns:
                category_column = col
                break
        
        # Strategy 2: If no exact match, use the second column (most common pattern)
        if not category_column and len(data.columns) >= 2:
            second_col = data.columns[1]
            # Check if it's not a date column
            if not (second_col.count('-') >= 2 and any(char.isdigit() for char in second_col)):
                category_column = second_col
                print(f"Using second column as category: '{second_col}'")
        
        # Strategy 3: Find any non-date, non-app-name column
        if not category_column:
            for col in data.columns:
                if (col not in ['App Name', 'App_Name'] and 
                    not (col.count('-') >= 2 and any(char.isdigit() for char in col)) and
                    not col.replace('-', '').replace('/', '').isdigit()):
                    category_column = col
                    print(f"Found potential category column: '{col}'")
                    break
        
        # Extract categories
        if category_column:
            raw_categories = data[category_column].head(num_rows).fillna('Unknown').astype(str).tolist()
            categories = [cat.strip() if cat.strip() != 'nan' else 'Unknown' for cat in raw_categories]
            unique_cats = list(set([cat for cat in categories if cat != 'Unknown']))
            
            print(f"\n✅ SUCCESS: Found category column '{category_column}'")
            print(f"Categories extracted: {categories}")
            print(f"Unique categories: {unique_cats}")
            
            # Check if we got the expected categories
            expected_cats = ['INC', 'AAR', 'SSR', 'NSSR', 'CC']
            found_expected = [cat for cat in unique_cats if cat in expected_cats]
            if found_expected:
                print(f"🎉 GREAT: Found expected category types: {found_expected}")
            else:
                print(f"⚠️  Note: Categories found but not the expected INC/AAR/SSR types: {unique_cats}")
        else:
            print(f"❌ FAILED: No category column found")
            print(f"Available columns: {list(data.columns)}")
            
        # Show sample data for verification
        print(f"\nSample data (first 3 rows):")
        if 'App Name' in data.columns and category_column:
            sample_df = data[['App Name', category_column]].head(3)
            print(sample_df)
        else:
            print(data.head(3))
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_category_extraction()
