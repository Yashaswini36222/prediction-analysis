import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Tuple, Dict

from data_preprocessing import clean_excel_data
from feature_engineering import run_feature_engineering

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                 X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[XGBRegressor, Dict, Dict]:
    """
    Train XGBoost model on the pre-split data
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
    
    Returns:
        Tuple of (trained_model, train_metrics, val_metrics)
    """
    print("\n=== Training XGBoost Model ===")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Feature columns: {X_train.columns.tolist()[:10]}...")  # Show first 10 features
    
    # Initialize XGBoost model with optimized parameters
    # model = XGBRegressor(
    #     n_estimators=500,  # Reduced for faster training
    #     max_depth=6,
    #     learning_rate=0.1,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     early_stopping_rounds=20,
    #     eval_metric='rmse'
    # )
    model = XGBRegressor(
        n_estimators=1000,      # Increased for higher precision
        max_depth=5,            # Shallow trees to prevent overfitting on 2023 data
        learning_rate=0.01,     # Slow learning for 'perfect' convergence
        subsample=0.9,          # Use 90% of data for each tree
        colsample_bytree=0.9,
        gamma=0.1,              # Minimum loss reduction for a split
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae'       # Focus on 'Mean Absolute Error' for realistic numbers
    )
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)],
        verbose=False  # Show training progress
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    # Calculate MAPE (Mean Absolute Percentage Error)
    def calculate_mape(y_true, y_pred):
        # Avoid division by zero
        mask = y_true != 0
        if not any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred),
        'mape': calculate_mape(y_train, y_train_pred)
    }
    val_metrics = {
        'mse': mean_squared_error(y_val, y_val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'mae': mean_absolute_error(y_val, y_val_pred),
        'r2': r2_score(y_val, y_val_pred),
        'mape': calculate_mape(y_val, y_val_pred)
    }
    
    return model, train_metrics, val_metrics

def run_xgboost_training(df: pd.DataFrame, scalers: Dict, prediction_horizon: int = 3) -> Tuple[XGBRegressor, Dict, Dict]:
    """
    Run complete XGBoost training pipeline with multi-horizon support
    
    Args:
        df: Preprocessed DataFrame from data_preprocessing
        scalers: Scalers dictionary from data_preprocessing
        prediction_horizon: Number of months to forecast (3 or 6)
    
    Returns:
        Tuple of (trained_model, train_metrics, val_metrics)
    """
    try:
        print(f"\n=== XGBoost Training Pipeline ({prediction_horizon}-month horizon) ===")
        
        # Step 1: Feature Engineering with prediction horizon
        results = run_feature_engineering(df, scalers, prediction_horizon)
        
        if results is None:
            raise ValueError("Feature engineering failed!")
        
        (X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb,
         X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm,
         model_scalers) = results
        
        # Step 2: Train XGBoost Model
        model, train_metrics, val_metrics = train_xgboost(
            X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb
        )
        
        # Step 3: Display Results
        print("\n=== XGBoost Training Results ===")
        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Step 4: Save Model and Metrics
        if not os.path.exists('models'):
            os.makedirs('models')
        
        model_path = f'models/xgboost_model_{prediction_horizon}month.joblib'
        joblib.dump(model, model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save metrics
        if not os.path.exists('metrics'):
            os.makedirs('metrics')
        
        metrics_path = f'metrics/xgboost_metrics_{prediction_horizon}month.joblib'
        joblib.dump(val_metrics, metrics_path)
        print(f"✅ Metrics saved to: {metrics_path}")
        
        # Step 5: Feature Importance
        print("\nTop 10 Feature Importances:")
        feature_importance = pd.DataFrame({
            'feature': X_train_xgb.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model, train_metrics, val_metrics
        
    except Exception as e:
        print(f"\nError in XGBoost training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    try:
        # Step 1: Data Preprocessing
        file_path = "App wise actuals_2023 to 2024.xlsx"
        print("\n=== XGBoost Model Training ===")
        print(f"Processing file: {file_path}")
        
        df, scalers = clean_excel_data(file_path)
        
        if df is None:
            raise ValueError("Data preprocessing failed!")
        
        # Step 2: Run XGBoost Training
        model, train_metrics, val_metrics = run_xgboost_training(df, scalers)
        
        if model is not None:
            print("\n🎉 XGBoost training completed successfully!")
        else:
            print("\n❌ XGBoost training failed!")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
