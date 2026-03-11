import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import tensorflow as tf
from xgboost import XGBRegressor
import os
from typing import Dict, Tuple, List
from data_preprocessing import clean_excel_data
from xgboost_model import run_xgboost_training
from lstm_model import run_sliding_window_lstm_training

class EnsemblePredictor:
    def __init__(self, xgb_weight=0.5, lstm_weight=0.5):
        """
        Initialize ensemble model with weights for each model
        """
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.xgb_model = None
        self.lstm_model = None
        
    def load_models(self, xgb_path, lstm_path):
        """Load pre-trained models"""
        self.xgb_model = joblib.load(xgb_path)
        self.lstm_model = tf.keras.models.load_model(lstm_path)
        
    def predict(self, X_xgb, X_lstm):
        """
        Make predictions using both models and combine them
        Args:
            X_xgb: Features formatted for XGBoost
            X_lstm: Features formatted for LSTM (3D array)
        """
        xgb_pred = self.xgb_model.predict(X_xgb)
        lstm_pred = self.lstm_model.predict(X_lstm).flatten()
        
        # Weighted average of predictions
        ensemble_pred = (self.xgb_weight * xgb_pred + 
                        self.lstm_weight * lstm_pred)
        
        return ensemble_pred
    
    def evaluate(self, X_xgb, X_lstm, y_true):
        """
        Evaluate ensemble model performance
        
        Args:
            X_xgb: Features for XGBoost
            X_lstm: Features for LSTM
            y_true: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_xgb, X_lstm)
        
        metrics = {
            'mse': mean_squared_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'mae': mean_absolute_error(y_true, predictions),
            'r2': r2_score(y_true, predictions)
        }
        
        return metrics


class MultiHorizonEnsemble:
    """
    Multi-horizon ensemble system combining XGBoost and LSTM models
    """
    
    def __init__(self):
        self.models = {
            'xgboost': {},
            'lstm': {},
            'ensemble_weights': {}
        }
        self.performance_metrics = {}
    
    def calculate_adaptive_weights(self, xgb_r2: float, lstm_r2: float, horizon: int) -> Tuple[float, float]:
        """
        Calculate adaptive weights based on model performance
        """
        print(f"\n🎯 Calculating weights for {horizon}-month horizon:")
        print(f"   XGBoost R²: {xgb_r2:.4f}")
        print(f"   LSTM R²: {lstm_r2:.4f}")
        
        # Performance-based weighting
        min_weight = 0.1  # Minimum weight for any model
        
        if xgb_r2 <= 0 and lstm_r2 <= 0:
            # Both models poor - equal weights
            xgb_weight, lstm_weight = 0.5, 0.5
        elif xgb_r2 <= 0:
            # Only LSTM is good
            xgb_weight, lstm_weight = min_weight, 1 - min_weight
        elif lstm_r2 <= 0:
            # Only XGBoost is good
            xgb_weight, lstm_weight = 1 - min_weight, min_weight
        else:
            # Both models are good - weight by relative performance
            total_performance = xgb_r2 + lstm_r2
            xgb_weight = xgb_r2 / total_performance
            lstm_weight = lstm_r2 / total_performance
            
            # Ensure minimum weights
            if xgb_weight < min_weight:
                xgb_weight = min_weight
                lstm_weight = 1 - min_weight
            elif lstm_weight < min_weight:
                lstm_weight = min_weight
                xgb_weight = 1 - min_weight
        
        print(f"   Calculated weights: XGBoost={xgb_weight:.3f}, LSTM={lstm_weight:.3f}")
        return xgb_weight, lstm_weight
    
    def train_ensemble_models(self, df: pd.DataFrame, scalers: Dict, 
                            horizons: List[int] = [3, 6]) -> Dict:
        """
        Train both XGBoost and LSTM models for specified horizons
        """
        results = {
            'xgboost': {},
            'lstm': {},
            'ensemble_weights': {},
            'performance': {}
        }
        
        print("\n" + "="*70)
        print("🚀 ENSEMBLE MODEL TRAINING")
        print("="*70)
        
        for horizon in horizons:
            print(f"\n📊 Training models for {horizon}-month horizon...")
            
            # Train XGBoost
            try:
                print(f"\n🌟 Training XGBoost ({horizon} months)...")
                xgb_model, xgb_train_metrics, xgb_val_metrics = run_xgboost_training(
                    df, scalers, prediction_horizon=horizon
                )
                
                results['xgboost'][horizon] = {
                    'model': xgb_model,
                    'train_metrics': xgb_train_metrics,
                    'val_metrics': xgb_val_metrics
                }
                
                print(f"✅ XGBoost {horizon}-month: R² = {xgb_val_metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"❌ XGBoost {horizon}-month training failed: {str(e)}")
                results['xgboost'][horizon] = None
            
            # Train LSTM
            try:
                print(f"\n🧠 Training LSTM ({horizon} months)...")
                lstm_model, lstm_train_metrics, lstm_val_metrics = run_sliding_window_lstm_training(
                    df, scalers, prediction_length=horizon
                )
                
                results['lstm'][horizon] = {
                    'model': lstm_model,
                    'train_metrics': lstm_train_metrics,
                    'val_metrics': lstm_val_metrics
                }
                
                lstm_r2 = lstm_val_metrics.get('overall', {}).get('r2', 0.0)
                print(f"✅ LSTM {horizon}-month: R² = {lstm_r2:.4f}")
                
            except Exception as e:
                print(f"❌ LSTM {horizon}-month training failed: {str(e)}")
                results['lstm'][horizon] = None
            
            # Calculate ensemble weights
            if results['xgboost'][horizon] and results['lstm'][horizon]:
                xgb_r2 = results['xgboost'][horizon]['val_metrics']['r2']
                lstm_r2 = results['lstm'][horizon]['val_metrics'].get('overall', {}).get('r2', 0.0)
                
                xgb_weight, lstm_weight = self.calculate_adaptive_weights(xgb_r2, lstm_r2, horizon)
                
                results['ensemble_weights'][horizon] = {
                    'xgboost': xgb_weight,
                    'lstm': lstm_weight
                }
                
                # Store in class for later use
                self.models['xgboost'][horizon] = results['xgboost'][horizon]['model']
                self.models['lstm'][horizon] = results['lstm'][horizon]['model']
                self.models['ensemble_weights'][horizon] = results['ensemble_weights'][horizon]
                
            else:
                print(f"⚠️ Cannot create ensemble for {horizon}-month (missing models)")
        
        return results
    
    def evaluate_ensemble_performance(self, results: Dict, horizons: List[int]) -> Dict:
        """
        Evaluate and compare ensemble performance
        """
        evaluation = {}
        
        print("\n" + "="*70)
        print("📊 ENSEMBLE PERFORMANCE EVALUATION")
        print("="*70)
        
        for horizon in horizons:
            if horizon not in results['ensemble_weights']:
                continue
                
            print(f"\n🎯 {horizon}-MONTH FORECASTING EVALUATION:")
            print("-" * 50)
            
            # Individual model performance
            xgb_metrics = results['xgboost'][horizon]['val_metrics']
            lstm_metrics = results['lstm'][horizon]['val_metrics']
            
            xgb_r2 = xgb_metrics['r2']
            lstm_r2 = lstm_metrics.get('overall', {}).get('r2', lstm_metrics.get('r2', 0.0))
            
            # Ensemble weights
            weights = results['ensemble_weights'][horizon]
            
            # Theoretical ensemble performance (weighted average)
            ensemble_r2_theoretical = (weights['xgboost'] * xgb_r2 + 
                                     weights['lstm'] * lstm_r2)
            
            evaluation[horizon] = {
                'xgboost': {
                    'r2': xgb_r2,
                    'rmse': xgb_metrics['rmse'],
                    'mae': xgb_metrics['mae'],
                    'weight': weights['xgboost']
                },
                'lstm': {
                    'r2': lstm_r2,
                    'rmse': lstm_metrics.get('overall', {}).get('rmse', lstm_metrics.get('rmse', 0.0)),
                    'mae': lstm_metrics.get('overall', {}).get('mae', lstm_metrics.get('mae', 0.0)),
                    'weight': weights['lstm']
                },
                'ensemble_theoretical': {
                    'r2': ensemble_r2_theoretical,
                    'weights': weights
                }
            }
            
            # Display results
            print(f"🌟 XGBoost: R² = {xgb_r2:.4f} (weight: {weights['xgboost']:.3f})")
            print(f"🧠 LSTM: R² = {lstm_r2:.4f} (weight: {weights['lstm']:.3f})")
            print(f"🎯 Ensemble (theoretical): R² = {ensemble_r2_theoretical:.4f}")
            
            # Determine best individual model
            if xgb_r2 > lstm_r2:
                best_individual = "XGBoost"
                improvement = ensemble_r2_theoretical - xgb_r2
            else:
                best_individual = "LSTM"
                improvement = ensemble_r2_theoretical - lstm_r2
            
            print(f"🏆 Best individual: {best_individual}")
            print(f"📈 Ensemble improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        return evaluation
    
    def save_ensemble_models(self, results: Dict, horizons: List[int]):
        """Save all ensemble models and weights"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        print("\n💾 Saving ensemble models...")
        
        for horizon in horizons:
            if horizon in results['ensemble_weights']:
                # CHANGE: Save as 'ensemble_12month.joblib' so the API sees it as a model
                weights_path = f'models/ensemble_{horizon}month.joblib'
                joblib.dump(results['ensemble_weights'][horizon], weights_path)
                print(f"✅ Ensemble model saved: {weights_path}")
        
        # Save ensemble configuration
        ensemble_config = {
            'horizons': horizons,
            'weights': results['ensemble_weights'],
            'model_paths': {
                'xgboost': {h: f'models/xgboost_model_{h}month.joblib' for h in horizons},
                'lstm': {h: f'models/sliding_window_lstm_{h}month.h5' for h in horizons}
            }
        }
        
        config_path = 'models/ensemble_config.joblib'
        joblib.dump(ensemble_config, config_path)
        print(f"✅ Ensemble configuration saved: {config_path}")


def run_ensemble_training(df: pd.DataFrame, scalers: Dict, 
                         horizons: List[int] = [3, 6,12]) -> Tuple[MultiHorizonEnsemble, Dict, Dict]:
    """
    Run complete ensemble training pipeline
    
    Args:
        df: Preprocessed DataFrame
        scalers: Scalers from preprocessing
        horizons: List of forecasting horizons
        
    Returns:
        Tuple of (ensemble_system, results, evaluation)
    """
    # Initialize ensemble system
    ensemble = MultiHorizonEnsemble()
    
    # Train all models
    results = ensemble.train_ensemble_models(df, scalers, horizons)
    
    # Evaluate ensemble performance
    evaluation = ensemble.evaluate_ensemble_performance(results, horizons)
    
    # Save models
    ensemble.save_ensemble_models(results, horizons)
    
    # Generate summary report
    print("\n" + "="*80)
    print("📋 ENSEMBLE TRAINING SUMMARY")
    print("="*80)
    
    total_models = len(horizons) * 2  # XGBoost + LSTM for each horizon
    successful_models = sum([
        1 for horizon in horizons 
        for model_type in ['xgboost', 'lstm']
        if results[model_type].get(horizon) is not None
    ])
    
    successful_ensembles = len([h for h in horizons if h in results['ensemble_weights']])
    
    print(f"✅ Successfully trained: {successful_models}/{total_models} individual models")
    print(f"✅ Successfully created: {successful_ensembles}/{len(horizons)} ensemble models")
    print(f"📁 All models saved in: ./models/ directory")
    
    # Performance summary
    if successful_ensembles > 0:
        avg_improvement = np.mean([
            evaluation[h]['ensemble_theoretical']['r2'] - max(
                evaluation[h]['xgboost']['r2'], 
                evaluation[h]['lstm']['r2']
            ) for h in horizons if h in evaluation
        ])
        
        print(f"📈 Average ensemble improvement: {avg_improvement:+.4f} ({avg_improvement*100:+.2f}%)")
        
        if avg_improvement > 0.01:
            print("🎉 Ensemble provides significant performance benefits!")
        elif avg_improvement > 0:
            print("⚡ Ensemble provides marginal performance benefits")
        else:
            print("⚠️ Individual models may be sufficient")
    
    return ensemble, results, evaluation


def prepare_ensemble_data(df, scalers, sequence_length=12):
    """
    Prepare data for both XGBoost and LSTM models
    """
    from feature_engineering import create_time_series_features, prepare_data_for_modeling
    
    # Get features for both models
    df_features = create_time_series_features(df)
    
    return prepare_data_for_modeling(df_features, sequence_length, scalers)

def train_ensemble(X_train_xgb, X_train_lstm, y_train, 
                  X_val_xgb, X_val_lstm, y_val,
                  xgb_params=None, lstm_params=None):
    """
    Train both models and create ensemble
    """
    # Default XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 1000,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42
        }
    
    # Train XGBoost
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_xgb, y_train,
                  eval_set=[(X_val_xgb, y_val)],
                  early_stopping_rounds=50,
                  verbose=False)
    
    # Train LSTM
    if lstm_params is None:
        lstm_model = build_lstm_model(X_train_lstm.shape[1:])
    else:
        lstm_model = build_lstm_model(X_train_lstm.shape[1:], **lstm_params)
    
    lstm_model.fit(X_train_lstm, y_train,
                   validation_data=(X_val_lstm, y_val),
                   epochs=100,
                   batch_size=32,
                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
    
    return xgb_model, lstm_model

def build_lstm_model(input_shape, units=64, dropout=0.2):
    """Build LSTM model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units//2),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    """
    Main execution function for ensemble training
    """
    print("🎯 AIMS Incident Trends - Ensemble Model Training")
    print("="*70)
    
    # Default data file path
    default_file = "App wise actuals_2023 to 2024.xlsx"
    
    try:
        # Load and preprocess data
        print("\n🔄 Loading and preprocessing data...")
        df, scalers = clean_excel_data(default_file)
        
        if df is None:
            raise ValueError("Data preprocessing failed!")
        
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Run ensemble training for both 3 and 6 month horizons
        horizons = [3, 6]
        ensemble, results, evaluation = run_ensemble_training(df, scalers, horizons)
        
        print("\n🎉 Ensemble training completed successfully!")
        print("\n📁 All models and reports saved in ./models/ directory")
        
        # Display final performance summary
        print("\n" + "="*80)
        print("🏆 FINAL PERFORMANCE SUMMARY")
        print("="*80)
        
        for horizon in horizons:
            if horizon in evaluation:
                eval_data = evaluation[horizon]
                print(f"\n📊 {horizon}-Month Forecasting:")
                print(f"   🌟 XGBoost: R² = {eval_data['xgboost']['r2']:.4f}")
                print(f"   🧠 LSTM: R² = {eval_data['lstm']['r2']:.4f}")
                print(f"   🎯 Ensemble: R² = {eval_data['ensemble_theoretical']['r2']:.4f}")
                
                best_r2 = max(eval_data['xgboost']['r2'], eval_data['lstm']['r2'])
                ensemble_r2 = eval_data['ensemble_theoretical']['r2']
                improvement = ensemble_r2 - best_r2
                
                if improvement > 0:
                    print(f"   📈 Improvement: +{improvement:.4f} ({improvement*100:+.2f}%)")
                else:
                    print(f"   📉 Change: {improvement:.4f} ({improvement*100:+.2f}%)")
        
        print("\n🎯 Ensemble models are ready for production use!")
        
    except FileNotFoundError:
        print(f"❌ Data file '{default_file}' not found!")
        print("Please ensure the Excel file is in the current directory.")
    except Exception as e:
        print(f"❌ Ensemble training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
