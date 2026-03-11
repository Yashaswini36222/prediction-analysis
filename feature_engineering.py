"""
Feature Engineering Module for AIMS Incident Prediction System

This module provides functions for creating time-series features
and preparing data for different models.
"""

import re
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def run_feature_engineering(df, scalers=None, prediction_horizon=3):
    """
    Main feature engineering pipeline for model training.
    
    Args:
        df: Preprocessed DataFrame or path to data
        scalers: Scalers dictionary from data_preprocessing
        prediction_horizon: Number of months to forecast (3 or 6)
        
    Returns:
        Tuple of (X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb,
                 X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm,
                 model_scalers)
    """
    # Handle both DataFrame and path inputs
    if isinstance(df, str):
        df = pd.read_csv(df)
    else:
        df = df.copy()
    
    # Create time series features
    df_features = create_time_series_features(df)
    
    # Get month columns for time series data
    month_cols = [col for col in df.columns if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})', str(col).lower())]
    
    if not month_cols:
        month_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                     and col not in ['App Name', 'AIMS Tkt Category']]
        
    # 2. Create base features
    df_features = create_time_series_features(df)
    
    # Prepare data for XGBoost
    feature_cols = ['last_value', 'mean_last_3', 'mean_last_6', 'std_last_3', 'trend', 'holiday_season']
    available_features = [col for col in feature_cols if col in df_features.columns]
    
    if not available_features:
        # Create dummy features for demo
        df_features['feature_1'] = np.random.randn(len(df_features))
        df_features['feature_2'] = np.random.randn(len(df_features))
        available_features = ['feature_1', 'feature_2']
    
    # Create target variable (simplified for demo)
    if month_cols and len(month_cols) >= prediction_horizon:
        # Use the sum of next prediction_horizon months as target
        target_cols = month_cols[-prediction_horizon:]
        df_features['target'] = df_features[target_cols].sum(axis=1)
    else:
        # df_features['target'] = np.random.randn(len(df_features))
        df_features['target'] = 0
    
    # Split data (80/20 split)
    split_idx = int(len(df_features) * 0.8)
    
    # XGBoost data
    X_train_xgb = df_features[available_features][:split_idx].fillna(0)
    y_train_xgb = df_features['target'][:split_idx].fillna(0)
    X_val_xgb = df_features[available_features][split_idx:].fillna(0)
    y_val_xgb = df_features['target'][split_idx:].fillna(0)
    
    # LSTM data preparation
    sequence_length = min(12, len(month_cols)) if month_cols else 12
    X_lstm, y_lstm = prepare_lstm_data(df, sequence_length, prediction_horizon)
    
    # Split LSTM data
    lstm_split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm = X_lstm[:lstm_split_idx]
    y_train_lstm = y_lstm[:lstm_split_idx]
    X_val_lstm = X_lstm[lstm_split_idx:]
    y_val_lstm = y_lstm[lstm_split_idx:]
    
    # Model scalers (if not provided, create new ones)
    if scalers is None:
        scalers = {'scaler': MinMaxScaler()}
    
    return (X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb,
            X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm,
            scalers)

"""" old one need to test and compare with new one
def create_time_series_features(df, include_cyclical=True):
    
    Create time-series features from the data.
    
    Args:
        df: Input DataFrame
        include_cyclical: Whether to include cyclical features
        
    Returns:
        DataFrame with time-series features
    
    df = df.copy()
    
    # Get month columns (columns that represent months)
    month_cols = [col for col in df.columns if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})', str(col).lower())]
    month_cols = [c for c in month_cols if 'total' not in str(c).lower()]
    if not month_cols:
        # If no month columns found, look for date-like columns
        month_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                     and col not in ['App Name', 'AIMS Tkt Category']]
    
    # For each row, create lag features
    for idx in df.index:
        values = df.loc[idx, month_cols].values if month_cols else []
        
        # Create lag features
        if len(values) > 0:
            df.loc[idx, 'last_value'] = values[-1] if len(values) > 0 else 0
            df.loc[idx, 'mean_last_3'] = np.mean(values[-3:]) if len(values) >= 3 else np.mean(values)
            df.loc[idx, 'mean_last_6'] = np.mean(values[-6:]) if len(values) >= 6 else np.mean(values)
            df.loc[idx, 'std_last_3'] = np.std(values[-3:]) if len(values) >= 3 else 0
            df.loc[idx, 'trend'] = (values[-1] - values[-3]) / 3 if len(values) >= 3 else 0

            # Holiday Season logic (December/January)
            if month_cols:
                last_month_name = str(month_cols[-1]).lower()
                # If the current month is Dec or Jan, mark it as Holiday Season (1)
                df.loc[idx, 'holiday_season'] = 1 if any(m in last_month_name for m in ['dec', 'jan']) else 0

            if include_cyclical and len(month_cols) >= 12:
                df.loc[idx, 'seasonality'] = np.mean(values[::3]) if len(values) >= 3 else 0
    
    return df
"""
def create_time_series_features(df):
    """
    HYBRID AI ENHANCEMENT: Adds Momentum and Recency Weighting
    """
    df = df.copy()
    month_cols = [col for col in df.columns if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})', str(col).lower())]
    
    for idx in df.index:
        values = df.loc[idx, month_cols].values.astype(float)
        
        if len(values) >= 3:
            # 1. Base Features
            df.loc[idx, 'last_value'] = values[-1]
            df.loc[idx, 'mean_last_3'] = np.mean(values[-3:])
            
            # 2. MOMENTUM (The 'Perfect' Logic enhancement)
            # Detects if incidents are accelerating or slowing down
            df.loc[idx, 'momentum'] = (values[-1] - values[-3]) / 2
            
            # 3. RECENCY (Weighting 2024/2025 higher than 2023)
            weights = np.linspace(0.5, 1.0, len(values))
            df.loc[idx, 'weighted_trend'] = np.average(values, weights=weights)
            
            # 4. HOLIDAY SEASON FLAG
            # Marking Dec/Jan for the AI reasoning layer
            last_month = str(month_cols[-1]).lower()
            df.loc[idx, 'holiday_season'] = 1 if any(m in last_month for m in ['dec', 'jan']) else 0

    return df



def prepare_data_for_modeling(df, target_col=None, feature_cols=None, test_size=0.2):
    """
    Prepare data for model training.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: List of feature columns
        test_size: Test set size
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    df = df.copy()
    
    # If no specific columns provided, use engineered features
    if feature_cols is None:
        feature_cols = ['last_value', 'mean_last_3', 'mean_last_6', 'std_last_3', 'trend', 'holiday_season']
        # Add any additional numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # If no target column specified, create a dummy target
    if target_col is None:
        # For demo purposes, predict next value based on trend
        if 'last_value' in df.columns and 'trend' in df.columns:
            df['target'] = df['last_value'] + df['trend']
        else:
            df['target'] = 0
        target_col = 'target'
    
    # Ensure feature columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    if not available_features:
        # Create minimal features
        df['feature_1'] = np.random.randn(len(df))
        df['feature_2'] = np.random.randn(len(df))
        available_features = ['feature_1', 'feature_2']
    
    # Prepare X and y
    X = df[available_features].fillna(0)
    y = df[target_col].fillna(0) if target_col in df.columns else np.zeros(len(df))
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test, available_features


def prepare_lstm_data(df, sequence_length=12, forecast_horizon=3):
    """
    Prepare data specifically for LSTM models.
    
    Args:
        df: Input DataFrame
        sequence_length: Number of time steps to look back
        forecast_horizon: Number of steps to predict
        
    Returns:
        Tuple of (X, y) suitable for LSTM training
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # If we have month columns, use them
    # month_cols = [col for col in df.columns if any(month in str(col).lower() 
    #               for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
    #                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
    
    month_cols = [col for col in df.columns if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})', str(col).lower())]
    month_cols = [c for c in month_cols if 'offboarding' not in str(c).lower() and 'total' not in str(c).lower()]
    if month_cols:
        # Use month columns as time series
        # data = df[month_cols].values
        data = df[month_cols].values if month_cols else np.zeros((len(df), 12))
        X, y = [], []
    else:
        # Use any numeric data available
        data = df[numeric_cols].values if numeric_cols else np.random.randn(len(df), 12)
    
    # Create sequences for LSTM
    X, y = [], []
    
    # For each row, create sequences
    for i in range(len(data)):
        row_data = data[i]
        if len(row_data) >= sequence_length + forecast_horizon:
            for j in range(len(row_data) - sequence_length - forecast_horizon + 1):
                X.append(row_data[j:j+sequence_length])
                y.append(row_data[j+sequence_length:j+sequence_length+forecast_horizon])
    
    # If not enough data, create synthetic sequences
    if len(X) < 10:
        # Create synthetic data for demo
        for i in range(100):
            X.append(np.random.randn(sequence_length))
            y.append(np.random.randn(forecast_horizon))
    if not X: # Synthetic fallback
        return np.zeros((10, sequence_length, 1)), np.zeros((10, forecast_horizon))
    
    return np.array(X).reshape(len(X), sequence_length, 1), np.array(y)

def add_seasonal_features(df: pd.DataFrame, month_columns: List[str]):
    """Adds a 'Holiday_Season' multiplier to help the AI learn winter drops."""
    for col in month_columns:
        # Extract month from column name (e.g., 'Jan-25' -> 1)
        # This works for 2023, 2024, and 2025
        month_name = str(col).lower()
        
        is_holiday = any(m in month_name for m in ['dec', 'jan'])
        
        # We create a 'Seasonality_Weight' feature
        # 1.0 = Normal, 0.7 = Expected 30% drop during holidays
        df[f'{col}_seasonal_weight'] = 0.7 if is_holiday else 1.0
        
    return df


def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        # Return uniform importance if not available
        importance = np.ones(len(feature_names)) / len(feature_names)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


# Backward compatibility functions
def engineer_features(df):
    """Backward compatibility wrapper"""
    return create_time_series_features(df)


def prepare_modeling_data(df):
    """Backward compatibility wrapper"""
    return prepare_data_for_modeling(df) 

