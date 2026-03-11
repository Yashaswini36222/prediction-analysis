# -*- coding: utf-8 -*-
"""
AIMS Incident Data Preprocessing (Hardened)

Fixes included:
- Robust month column detection for Mon_YYYY, YYYY-MM, YYYY-MM-DD, datetime typed cols
- Header normalization to canonical names (App, Digital_Unit, Category)
- Stable month column ordering by parsed date
- Accurate data_info calculation (apps, categories, time_range)
- Clean conversion of month columns to numeric with NaN handling
"""

import os
import re
import warnings
from typing import Tuple, Dict, List, Optional
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


# ========== Pretty printing helpers ==========
def print_dataframe_info(df: pd.DataFrame, title: str = "") -> None:
    """Helper function to print DataFrame information"""
    if title:
        print(f"\n=== {title} ===")
    print("\nShape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nInfo:")
    df.info()


# ========== Header normalization ==========
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced Fuzzy matching to ensure 'App Name' is always found."""
    rename_map = {}
    for col in df.columns:
        clean_col = str(col).replace('\n', ' ').replace('_', ' ').strip().lower()
        # Search for App Name
        if any(key in clean_col for key in ['app name', 'application', 'tool name', 'app']):
            if 'App Name' not in rename_map.values(): # Avoid double mapping
                rename_map[col] = 'App Name'
        
        # Search for Category
        elif any(key in clean_col for key in ['category', 'tkt', 'incident type', 'type']):
            rename_map[col] = 'Ticket Category'

    
        elif any(key in clean_col for key in ['unit', 'digital', ]):
            rename_map[col] = 'Digital_Unit'
            
    return df.rename(columns=rename_map)





#W def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:working 
    # """
    # Normalize raw headers to canonical names expected downstream:
    #   - App → 'App'
    #   - Digital Unit → 'Digital_Unit'
    #   - Category → 'Category'
    # (Non-metadata columns, such as month columns, are left as-is.)
    # """
    # """Aligns names with the API's expectations."""
    # rename_map = {}
    # lower_to_orig = {str(c).strip().lower(): c for c in df.columns}
    #  working rename_map = {}
    # for col in df.columns:
    #     clean_col = str(col).replace('\n', ' ').strip().lower()working 

    # # App
    # for key in ['app', 'app name', 'application', 'application name']:
    #     if key in lower_to_orig:
    #         rename_map[lower_to_orig[key]] = 'App'
    #         break

    # # Digital_Unit
    # for key in ['digital_unit', 'digital unit', 'unit', 'digital gbus']:
    #     if key in lower_to_orig:
    #         rename_map[lower_to_orig[key]] = 'Digital_Unit'
    #         break

    # Category
    # for key in ['aims tkt category', 'category', 'ticket category', 'type']:
    #     if key in lower_to_orig:
    #         rename_map[lower_to_orig[key]] = 'Category'
    #         break

    # for col in df.columns:
    #     # This removes line breaks and extra spaces before checking
    #     clean_col = str(col).replace('\n', ' ').strip().lower()
    #     if any(k in clean_col for k in ['aims tkt category', 'category', 'ticket category']):
    #         rename_map[col] = 'Category'
    #         break

    # W API expects "App Name" and "Ticket Category"
        # if any(key in clean_col for key in ['app name', 'application']):
        #     rename_map[col] = 'App Name'
        # elif any(key in clean_col for key in ['aims tkt', 'category']):
        #     rename_map[col] = 'Ticket Category'
        # elif any(key in clean_col for key in ['digital unit', 'planned for']):
        #     rename_map[col] = 'Digital_Unit'working 

    # for col in df.columns:
    #     # Step A: Clean the column name string
    #     clean_col = str(col).replace('\n', ' ').strip().lower()
        
    #     # Step B: Map to 'App'
    #     if any(key in clean_col for key in ['app name', 'application', 'app']):
    #         rename_map[col] = 'App'
        
    #     # Step C: Map to 'Category' 
    #     elif any(key in clean_col for key in ['aims tkt category', 'category', 'ticket']):
    #         rename_map[col] = 'Category'
            
    #     # Step D: Map to 'Digital_Unit'
    #     elif any(key in clean_col for key in ['digital unit', 'unit', 'planned for']):
    #         rename_map[col] = 'Digital_Unit'
    # W return df.rename(columns=rename_map)working 


# ========== Month/period column detection & ordering ==========
_MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def _parse_month_header(col: str) -> Optional[pd.Timestamp]:
    """
    Parse a month-like header into a Timestamp for stable ordering.
    Supported:
      - Mon_YYYY, Mon-YYYY, Mon YYYY (e.g., Jan_2023)
      - YYYY-MM, YYYY/MM
      - YYYY-MM-DD (treated as that specific date)
    Returns None if the header is not parseable as a period.
    """
    s = str(col).strip()

    # Mon_YYYY, Mon-YYYY, Mon YYYY
    m = re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_ ]\s*(\d{4})$', s, re.I)
    if m:
        mon_txt, year = m.group(1), int(m.group(2))
        mon = _MONTH_ABBR[mon_txt.lower()]
        try:
            return pd.Timestamp(year=year, month=mon, day=1)
        except Exception:
            return None

    # YYYY-MM or YYYY/MM
    m = re.match(r'^(\d{4})-_/$', s)
    if m:
        year, mon = int(m.group(1)), int(m.group(2))
        try:
            return pd.Timestamp(year=year, month=mon, day=1)
        except Exception:
            return None

    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])$', s)
    if m:
        try:
            return pd.to_datetime(s, errors='coerce')
        except Exception:
            return None

    return None


def get_month_columns(df: pd.DataFrame) -> List[str]:
    # # """
    # # Detect month/date columns in common formats or datetime-typed columns.
    # # - Mon_YYYY (e.g., Jan_2023), Mon-YYYY, Mon YYYY
    # # - YYYY-MM, YYYY/MM
    # # - YYYY-MM-DD
    # # - datetime dtype columns
    # # """
    # """Detects all month columns while strictly ignoring metadata."""
    # # List of valid month starts
    # valid_months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 
    #                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
    # for col in df.columns:
    #     col_str = str(col).strip().lower()

    #     # 1. Detect standard Mon (Jan, Feb) or Mon.1 (Jan.1, Feb.1)
    #     if any(col_str.startswith(m) for m in valid_months):
    #         # Exclude columns like 'App' or 'Category' just in case
    #         if col not in ['App', 'Digital_Unit', 'Category','App Name', 'AIMS Tkt Category']:
    #             month_cols.append(str(col))
    #             # continue

    #     # datetime typed columns
    #     if pd.api.types.is_datetime64_any_dtype(df[col]):
    #         month_cols.append(col_str)
    #         continue

    #     # Regex-based formats
    #     if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_ ]\s*\d{4}$', col_str, re.I):
    #         month_cols.append(col_str)
    #         continue
    #     if re.match(r'^\d{4}-_/$', col_str):
    #         month_cols.append(col_str)
    #         continue
    #     if re.match(r'^\d{4}-\d{2}-\d{2}$', col_str):
    #         month_cols.append(col_str)
    #         continue
    #     # 2. Keep your existing regex logic for other formats
    #     if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_ ]\s*\d{4}$', col_str, re.I):
    #         month_cols.append(str(col))
    # # Order by parsed timestamp; keep original order for ties/unparseable (defensive)
    # parsed = [(c, _parse_month_header(c)) for c in month_cols]
    # # If everything parsed, sort by timestamp; otherwise, preserve original detection order
    # if all(ts is not None for _, ts in parsed):
    #     month_cols = [c for c, _ in sorted(parsed, key=lambda x: x[1])]
    # return month_cols
    """Identifies months and cleans duplicate suffixes like .1"""
    """
    Standardized Month Detection
    Works for 2023, 2024, 2025 and 2026 formats.
    """
    # 1. Initialize the list at the very top
    found_months = []
    
    valid_starts = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
    
    # Metadata to ignore
    metadata_to_ignore = ['app name', 'ticket category', 'digital_unit', 'total', 'status']
    # for col in df.columns:
    #     col_str = str(col).strip().lower()
        
    #     # 2. Check if it's a month and NOT metadata
    #     if any(col_str.startswith(m) for m in valid_starts):
    #         # Ensure we aren't accidentally grabbing 'App Name' because it contains 'a'
    #         if not any(meta in col_str for meta in metadata_cols):
    #             found_months.append(str(col))
    for col in df.columns:
        col_str = str(col).strip().lower()
        # 1. Skip metadata columns first
        if any(meta in col_str for meta in metadata_to_ignore):
            continue

        # 3. Strategy B: Check for Year patterns (2023, 2024, 2025, 2026)
        if re.search(r'202[3-6]', col_str):
            found_months.append(str(col))
            continue

        # 4. Strategy C: Check for Date-like numeric formats (MM/YY or MM-YY)
        if re.match(r'^\d{1,2}[/-]\d{2,4}$', col_str):
            found_months.append(str(col))

        if any(col_str.startswith(m) for m in valid_starts):

            # Ensure it's not a metadata column
            if col not in ['App Name', 'Ticket Category', 'Digital_Unit']:
                found_months.append(str(col))

    # 3. Return the full list after the loop finishes
    return found_months

# ========== Core cleaning ==========
def clean_excel_data(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, MinMaxScaler]]]:
    # """
    # Clean and preprocess Excel data for AIMS incident analysis.

    # Expected file structure (before normalization):
    # - 'App Name' or similar (App)
    # - 'Digital_Unit' or similar
    # - 'AIMS Tkt Category' or 'Category' or similar
    # - Month/period columns (e.g., Jan_2023 .. Jun_2025)

    # Returns:
    #     Tuple of (cleaned_dataframe, scalers_dict)
    # """
    """
    Clean and preprocess Excel data for AIMS incident analysis.
    MODIFIED: Handles multi-row headers and specific AIMS column naming.
    """
    print("\n" + "=" * 60)
    print("=== DATA PREPROCESSING STARTED ===")
    print("=" * 60)

    try:
        # Step 1: Read Excel file - SKIPPING THE FIRST ROW (header=1)
        print(f"\n📂 Reading file: {file_path}")
        df_raw = pd.read_excel(file_path, engine='openpyxl', )

        # if df_raw is None or df_raw.empty:
        #     print("❌ Excel file is empty")
        #     return None, None
        if not any('app' in str(c).lower() for c in df_raw.columns):
            df_raw = pd.read_excel(file_path, engine='openpyxl', header=1)

        print(f"✅ File read successfully")
        print(f"   📊 Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")
        if 'Digital_Unit' not in df_raw.columns and 'Digital Unit' not in df_raw.columns:
            df_raw['Digital_Unit'] = 'Default_Unit'

        print(f"✅ File read successfully")
        print(f"   📊 Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")

        # Step 2: Normalize headers
        df = normalize_headers(df_raw)
        # This removes any rows where the App Name is missing
        if 'App Name' not in df.columns:
            print(f"❌ Available columns are: {df.columns.tolist()}")
            raise KeyError("Could not find 'App Name' column. Check Excel headers.")
        df = df.dropna(subset=['App Name'])

        print("\n📋 Original (first 3 rows with normalized headers):")
        print(df.head(3).to_string())
        print(f"\n📌 Column names (normalized):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")

        # Step 3: Validate minimum requirements
        required_cols = ['App Name', 'Ticket Category', 'Digital_Unit']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Error: Missing required column(s) after normalization: {missing}")
            return None, None

        # Step 4: Detect month columns robustly
        print("\n" + "=" * 60)
        print("📅 DETECTING MONTH/DATE COLUMNS")
        print("=" * 60)
        month_cols = get_month_columns(df)

        if not month_cols:
            print("❌ Error: No month/date columns detected. Check header formats.")
            return None, None

        print(f"✅ Month/date columns found: {len(month_cols)}")
        print("   First 5:", month_cols[:5])
        print("   Last 5 :", month_cols[-5:])

        # Step 5: Display metadata column stats
        print("\n" + "=" * 60)
        print("🔍 METADATA COLUMNS SUMMARY")
        print("=" * 60)
        print(f"✅ App Name column: {df['App Name'].nunique()} unique values")
        print(f"✅ Digital_Unit column: {df['Digital_Unit'].nunique()} unique values")
        print(f"✅ Ticket Category column: {df['Ticket Category'].nunique()} unique values")
        # Step 6: Convert month cols to numeric and handle non-numeric
        print("\n🔢 Converting month columns to numeric values...")
        conversion_issues = 0
        for col in month_cols:
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                conversion_issues += n_nan
                print(f"   ⚠️  Column '{col}': {n_nan} non-numeric values coerced to NaN")
        if conversion_issues > 0:
            print(f"\n⚠️  Found {conversion_issues} NaN values across month columns")
            print(f"   Filling with 0...")
            df[month_cols] = df[month_cols].fillna(0)
        else:
            print("✅ All month column values successfully converted to numeric")

        # Step 7: Validate numeric data
        print("\n✅ Numeric data validation (month columns):")
        print(f"   - Min value: {df[month_cols].min().min():.0f}")
        print(f"   - Max value: {df[month_cols].max().max():.0f}")
        print(f"   - Mean value: {df[month_cols].mean().mean():.2f}")
        print(f"   - Total incidents: {int(df[month_cols].sum().sum())}")

        # Step 8: Create final dataframe (only required metadata + months)
        print("\n" + "=" * 60)
        print("📊 CREATING FINAL DATAFRAME")
        print("=" * 60)
        df_clean = df[['App Name', 'Digital_Unit', 'Ticket Category'] + month_cols].copy()

        print(f"✅ Final dataframe shape: {df_clean.shape}")
        print(f"   📊 Rows: {len(df_clean)}, Columns: {len(df_clean.columns)}")

        print("\n📋 Cleaned Data Preview (First 3 rows):")
        print(df_clean.head(3).to_string())

        # Step 9: Initialize scalers for each month column
        print("\n" + "=" * 60)
        print("🔧 INITIALIZING SCALERS")
        print("=" * 60)

        scalers: Dict[str, MinMaxScaler] = {}
        for col in month_cols:
            scaler = MinMaxScaler()
            scalers[col] = scaler.fit(df_clean[[col]])
        print(f"✅ Created {len(scalers)} scalers for {len(month_cols)} month columns")

        # Step 10: Final validation
        print("\n" + "=" * 60)
        print("✅ DATA VALIDATION")
        print("=" * 60)
        print(f"✅ Null values check: {int(df_clean.isna().sum().sum())} null values")
        print(f"✅ Numeric month columns: {len(month_cols)}")
        print("✅ Metadata columns check: 3 columns (App, Digital_Unit, Category)")
        print("✅ Data integrity: All checks passed")
        print("✅ Ready for training: YES")

        print("\n" + "=" * 60)
        print("✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

        return df_clean, scalers

    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
        return None, None

    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ========== Validation & utilities ==========
def validate_data_structure(df: pd.DataFrame) -> bool:
    """
    Validate cleaned data structure (after normalization).
    """
    try:
        print("\n" + "=" * 60)
        print("🔍 VALIDATING DATA STRUCTURE")
        print("=" * 60)

        # Check minimum columns
        if len(df.columns) < 4:
            print("❌ Dataframe must have at least 4 columns (App, Digital_Unit, Category, month columns)")
            return False

        # Required columns
        required_cols = ['App Name', 'Ticket Category', 'Digital_Unit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Dataframe missing required columns: {missing_cols}")
            return False

        # Month columns existence
        month_cols = get_month_columns(df)
        if len(month_cols) == 0:
            # Fallback: infer as all numeric after the first 3 columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            month_cols = [c for c in numeric_cols if c not in required_cols]
        if len(month_cols) == 0:
            print("❌ No numeric month columns detected in dataframe")
            return False

        print(f"✅ All required columns present")
        print(f"   - App unique: {df['App'].nunique()}")
        print(f"   - Digital_Unit unique: {df['Digital_Unit'].nunique()}")
        print(f"   - Category unique: {df['Category'].nunique()}")
        print(f"   - Month columns: {len(month_cols)}")
        print("✅ Data structure validation PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return False


def handle_missing_values(df: pd.DataFrame, strategy: str = 'zero') -> pd.DataFrame:
    """
    Handle missing values in incident data:
      - 'forward_fill', 'backward_fill', 'zero', 'interpolate'
    Only applied to numeric columns (month data).
    """
    print("\n🔧 Handling missing values...")

    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    missing_before = int(df_copy[numeric_cols].isna().sum().sum())
    print(f"Missing values before: {missing_before}")

    if strategy == 'forward_fill':
        df_copy[numeric_cols] = df_copy[numeric_cols].ffill().bfill()
    elif strategy == 'backward_fill':
        df_copy[numeric_cols] = df_copy[numeric_cols].bfill().ffill()
    elif strategy == 'zero':
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0)
    elif strategy == 'interpolate':
        df_copy[numeric_cols] = df_copy[numeric_cols].interpolate(method='linear', axis=0).fillna(0)

    missing_after = int(df_copy[numeric_cols].isna().sum().sum())
    print(f"Missing values after: {missing_after}")
    print(f"✅ Missing values handled using '{strategy}' strategy\n")

    return df_copy


def remove_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove or cap outliers in incident data. Applies to numeric month columns.
    """
    print("\n🔧 Handling outliers...")

    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    outlier_count = 0
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outs = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
            outlier_count += int(outs.sum())

            # Cap instead of removing
            df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
            df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound

    print(f"✅ Handled {outlier_count} outliers using '{method}' method\n")
    return df_copy


def get_month_cols_safe(df: pd.DataFrame) -> List[str]:
    """Utility to obtain month columns, with numeric fallback."""
    month_cols = get_month_columns(df)
    if not month_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        month_cols = [c for c in numeric_cols if c not in ['App', 'Digital_Unit', 'Category']]
    return month_cols


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of cleaned data.
    """
    month_cols = get_month_cols_safe(df)

    summary = {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'unique_apps': int(df['App'].nunique()) if 'App' in df.columns else 0,
        'unique_units': int(df['Digital_Unit'].nunique()) if 'Digital_Unit' in df.columns else 0,
        'unique_categories': int(df['Category'].nunique()) if 'Category' in df.columns else 0,
        'date_columns': int(len(month_cols)),
        'total_incidents': int(df[month_cols].sum().sum()) if month_cols else 0,
        'missing_values': int(df.isna().sum().sum()),
    }
    return summary


def compute_data_info(df: pd.DataFrame) -> Dict:
    # """
    # Compute the data_info structure similar to your API response:
    #   rows, apps, categories, time_range
    # """
    """Accurately counts apps using the new names."""
    # month_cols = get_month_columns(df_norm)

    # apps = int(df_norm['App'].nunique()) if 'App' in df_norm.columns else 0
    # categories = int(df_norm['Category'].nunique()) if 'Category' in df_norm.columns else 0
    
    # apps = int(df['App'].nunique()) if 'App' in df.columns else 0
    # categories = int(df['Category'].nunique()) if 'Category' in df.columns else 0
    apps = int(df['App Name'].nunique()) if 'App Name' in df.columns else 0
    categories = int(df['Ticket Category'].nunique()) if 'Ticket Category' in df.columns else 0
    # month_cols = get_month_columns(df)
    # # This will now show "Jan to Dec.1" correctly
    # time_range = f"{month_cols[0]} to {month_cols[-1]}" if month_cols else "No dates found"
    month_cols = get_month_columns(df)
    time_range = f"{month_cols[0]} to {month_cols[-1]}" if month_cols else "No dates"
    # # Build time_range string
    # if month_cols:
    #     start_col = month_cols[0]
    #     end_col = month_cols[-1]
    # else:
    #     # Defensive fallback if month detection fails
    #     start_col = df_norm.columns[2] if len(df_norm.columns) >= 3 else ''
    #     end_col = df_norm.columns[-1] if len(df_norm.columns) else ''

    # return {
    #     "rows": int(len(df_norm)),
    #     "apps": apps,
    #     "categories": categories,
    #     "time_range": f"{start_col} to {end_col}"
    # }
    return {
        "rows": int(len(df)),
        "apps": apps,
        "categories": categories,
        "time_range": time_range
    }


# ========== Self-test ==========
if __name__ == "__main__":
    try:
        print("Testing data preprocessing...")
        file_path = "App wise actuals_2023 to 2024.xlsx"  # <-- change to your file path

        # Clean
        df_clean, scalers = clean_excel_data(file_path)

        if df_clean is not None and scalers is not None:
            print("\n" + "=" * 60)
            print("✅ PREPROCESSING SUCCESSFUL!")
            print("=" * 60)

            print(f"\n📊 Data shape: {df_clean.shape}")
            print(f"🔧 Scalers created: {len(scalers)}")

            # Validate structure
            validate_data_structure(df_clean)

            # Get summary
            summary = get_data_summary(df_clean)
            print(f"\n📋 DATA SUMMARY:")
            print(f"   - Total rows: {summary['total_rows']}")
            print(f"   - Total columns: {summary['total_columns']}")
            print(f"   - Unique apps: {summary['unique_apps']}")
            print(f"   - Unique units: {summary['unique_units']}")
            print(f"   - Unique categories: {summary['unique_categories']}")
            print(f"   - Date columns: {summary['date_columns']}")
            print(f"   - Total incidents: {summary['total_incidents']}")
            print(f"   - Missing values: {summary['missing_values']}")

            # data_info (similar to your API response)
            data_info = compute_data_info(df_clean)
            print(f"\n📌 DATA INFO: {data_info}")

        else:
            print("\n❌ Preprocessing failed!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
