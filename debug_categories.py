"""
Simple script to debug category column detection in uploaded Excel file
"""
import pandas as pd
import os

def debug_excel_structure():
    """Debug the structure of uploaded Excel files"""
    
    # Look for Excel files in the current directory
    excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
    
    if not excel_files:
        print("No Excel files found in current directory.")
        print("Please make sure your AIMS data Excel file is in this folder.")
        return
    
    print(f"Found Excel files: {excel_files}")
    
    # Analyze each Excel file
    for file in excel_files:
        print(f"\n{'='*50}")
        print(f"Analyzing: {file}")
        print('='*50)
        
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
            # Analyze each column
            print(f"\nColumn Analysis:")
            for i, col in enumerate(df.columns):
                print(f"{i}: '{col}'")
                if i < 5:  # Show details for first 5 columns
                    unique_vals = df[col].value_counts().head(10)
                    print(f"   Sample values: {list(unique_vals.index)}")
                    print(f"   Unique count: {df[col].nunique()}")
                
            # Try to identify category column
            print(f"\nCategory Column Detection:")
            
            # Check for exact matches
            exact_matches = ['Ticket Category', 'Category', 'Type', 'Incident Type', 'Ticket Type']
            found_exact = None
            for col in exact_matches:
                if col in df.columns:
                    found_exact = col
                    break
            
            if found_exact:
                print(f"Found exact match: '{found_exact}'")
                print(f"Values: {df[found_exact].value_counts().to_dict()}")
            else:
                print("No exact matches found")
                
                # Check second column
                if len(df.columns) >= 2:
                    second_col = df.columns[1]
                    print(f"Second column: '{second_col}'")
                    print(f"Values: {df[second_col].value_counts().head(10).to_dict()}")
                    
                    # Check if it contains category-like values
                    sample_values = df[second_col].dropna().astype(str).head(10).tolist()
                    print(f"Sample string values: {sample_values}")
                    
                    if any(val in ['INC', 'AAR', 'SSR', 'NSSR', 'CC'] for val in sample_values):
                        print("✅ This looks like the category column (contains INC/AAR/SSR/etc.)")
                    else:
                        print("❓ This might not be the category column")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    debug_excel_structure()
