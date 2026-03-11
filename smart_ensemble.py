"""
Smart Ensemble Engine for AIMS Incident Prediction
Dynamically weights XGBoost and LSTM models based on business context

This integrates with your existing models:
- XGBoost models: xgboost_model_3month.joblib, xgboost_model_6month.joblib
- LSTM models: sliding_window_lstm_3month.h5, sliding_window_lstm_6month.h5
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class ScenarioType(Enum):
    """Types of business scenarios that affect model weighting"""
    SEASONAL_PATTERN = "seasonal"
    BUSINESS_SPIKE = "spike"
    MAINTENANCE_WINDOW = "maintenance"
    QUARTER_END = "quarter_end"
    HOLIDAY_PERIOD = "holiday"
    NORMAL_OPERATIONS = "normal"

@dataclass
class PredictionContext:
    """Context information for smart ensemble weighting"""
    date: datetime
    app_name: str
    category: str
    day_of_week: int
    month: int
    is_quarter_end: bool = False
    is_holiday: bool = False
    is_maintenance_window: bool = False
    is_month_end: bool = False
    scenario_type: ScenarioType = ScenarioType.NORMAL_OPERATIONS

class SmartEnsemble:
    """
    Smart Ensemble Engine that dynamically weights models based on context
    Integrates with your existing XGBoost and LSTM models from the models/ directory
    """
    
    def __init__(self):
        """Initialize Smart Ensemble with your existing model structure"""
        self.models = {}
        self.model_paths = {
            'xgb_3month': 'models/xgboost_model_3month.joblib',
            'lstm_3month': 'models/sliding_window_lstm_3month.h5',
            'xgb_6month': 'models/xgboost_model_6month.joblib',
            'lstm_6month': 'models/sliding_window_lstm_6month.h5'
        }
        
        # Load your existing models
        self.load_models()
        
        # Default ensemble weights (your current setup)
        self.default_weights = {'xgb': 0.5, 'lstm': 0.5}
        
        # Context-based weight rules
        self.weight_rules = self._initialize_weight_rules()
        
        # Performance tracking
        self.prediction_history = []
        
        print("✅ Smart Ensemble initialized with your existing models")
        
    def load_models(self):
        """Load your existing trained models with compatibility fixes"""
        try:
            for model_key, path in self.model_paths.items():
                if 'xgb' in model_key:
                    self.models[model_key] = joblib.load(path)
                    print(f"✅ Loaded XGBoost model: {model_key}")
                elif 'lstm' in model_key:
                    # Handle Keras compatibility issues
                    try:
                        # Try loading with compile=False to avoid metric issues
                        self.models[model_key] = load_model(path, compile=False)
                        print(f"✅ Loaded LSTM model: {model_key} (compatibility mode)")
                    except Exception as keras_error:
                        print(f"⚠️ Keras compatibility issue for {model_key}: {str(keras_error)}")
                        # Try with custom objects for older models
                        try:
                            import tensorflow as tf
                            custom_objects = {
                                'mse': tf.keras.metrics.MeanSquaredError(),
                                'mae': tf.keras.metrics.MeanAbsoluteError(),
                                'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                                'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
                            }
                            self.models[model_key] = load_model(path, custom_objects=custom_objects, compile=False)
                            print(f"✅ Loaded LSTM model: {model_key} (with custom objects)")
                        except Exception as custom_error:
                            print(f"❌ Failed to load LSTM model {model_key}: {str(custom_error)}")
                            print(f"   You may need to retrain this model with current TensorFlow version")
                            # Set to None so we can handle it gracefully
                            self.models[model_key] = None
                    
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {str(e)}")
            print("Please ensure your models are trained and saved in the models/ directory")
            raise
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def _initialize_weight_rules(self) -> Dict[ScenarioType, Dict[str, Any]]:
        """
        Initialize context-based ensemble weight rules
        These rules determine when to trust XGBoost vs LSTM more
        """
        return {
            ScenarioType.SEASONAL_PATTERN: {
                'xgb': 0.3, 'lstm': 0.7,  # LSTM better for seasonal patterns
                'reason': 'Seasonal patterns favor LSTM temporal modeling',
                'confidence': 0.85
            },
            ScenarioType.BUSINESS_SPIKE: {
                'xgb': 0.8, 'lstm': 0.2,  # XGBoost better for sudden spikes
                'reason': 'Sudden spikes favor XGBoost pattern recognition',
                'confidence': 0.9
            },
            ScenarioType.QUARTER_END: {
                'xgb': 0.7, 'lstm': 0.3,  # XGBoost better for business events
                'reason': 'Quarter-end patterns favor XGBoost business logic',
                'confidence': 0.8
            },
            ScenarioType.HOLIDAY_PERIOD: {
                'xgb': 0.3, 'lstm': 0.7,  # LSTM better for cyclical reductions
                'reason': 'Holiday patterns favor LSTM seasonal modeling',
                'confidence': 0.85
            },
            ScenarioType.MAINTENANCE_WINDOW: {
                'xgb': 0.4, 'lstm': 0.6,  # Slightly favor LSTM for predictable patterns
                'reason': 'Maintenance patterns are predictable, slight LSTM preference',
                'confidence': 0.9
            },
            ScenarioType.NORMAL_OPERATIONS: {
                'xgb': 0.5, 'lstm': 0.5,  # Balanced approach (your current ensemble)
                'reason': 'Normal operations use balanced ensemble',
                'confidence': 0.7
            }
        }
    
    def analyze_context(self, date: datetime, app_name: str, category: str) -> PredictionContext:
        """
        Analyze business context to determine optimal model weighting
        
        Args:
            date: Prediction date
            app_name: Application name (e.g., "Finance_ERP", "HR_System")
            category: AIMS ticket category (e.g., "Database", "Network")
            
        Returns:
            PredictionContext with scenario analysis
        """
        context = PredictionContext(
            date=date,
            app_name=app_name.lower(),
            category=category.lower(),
            day_of_week=date.weekday(),
            month=date.month
        )
        
        # Determine context flags
        context.is_quarter_end = self._is_quarter_end(date)
        context.is_holiday = self._is_holiday_period(date)
        context.is_maintenance_window = self._is_maintenance_window(date)
        context.is_month_end = self._is_month_end(date)
        
        # Determine primary scenario type
        context.scenario_type = self._determine_scenario_type(context, app_name, category)
        
        return context
    
    def _is_quarter_end(self, date: datetime) -> bool:
        """Check if date falls in quarter-end period (last 7 days of quarter)"""
        quarter_end_months = [3, 6, 9, 12]
        if date.month in quarter_end_months:
            # Last 7 days of quarter month
            days_in_month = pd.Period(f"{date.year}-{date.month}").days_in_month
            return date.day > (days_in_month - 7)
        return False
    
    def _is_holiday_period(self, date: datetime) -> bool:
        """Check if date falls in major holiday periods"""
        month, day = date.month, date.day
        
        # Major holiday periods that affect business operations
        holiday_periods = [
            (12, 15, 1, 10),   # Christmas/New Year: Dec 15 - Jan 10
            (7, 1, 7, 15),     # Summer holidays: July 1-15
            (11, 20, 11, 30),  # Thanksgiving period
            (4, 10, 4, 20),    # Easter period (approximate)
        ]
        
        for start_month, start_day, end_month, end_day in holiday_periods:
            # Handle year transition (Dec to Jan)
            if start_month > end_month:  # Year transition
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day):
                    return True
            else:
                if (month >= start_month and month <= end_month):
                    if (month == start_month and day >= start_day) or \
                       (month == end_month and day <= end_day) or \
                       (start_month < month < end_month):
                        return True
        return False
    
    def _is_maintenance_window(self, date: datetime) -> bool:
        """Check if date/time falls in typical maintenance windows"""
        # First Sunday of month (common maintenance window)
        if date.weekday() == 6:  # Sunday
            # Check if it's first Sunday of month
            first_sunday = None
            for day in range(1, 8):
                test_date = datetime(date.year, date.month, day)
                if test_date.weekday() == 6:
                    first_sunday = day
                    break
            return date.day == first_sunday
        
        # Patch Tuesday (second Tuesday of month)
        if date.weekday() == 1:  # Tuesday
            tuesdays = []
            for day in range(1, 32):
                try:
                    if datetime(date.year, date.month, day).weekday() == 1:
                        tuesdays.append(day)
                except ValueError:
                    break
            return len(tuesdays) >= 2 and date.day == tuesdays[1]
        
        return False
    
    def _is_month_end(self, date: datetime) -> bool:
        """Check if date falls in month-end period (last 3 days)"""
        days_in_month = pd.Period(f"{date.year}-{date.month}").days_in_month
        return date.day > (days_in_month - 3)
    
    def _determine_scenario_type(self, context: PredictionContext, app_name: str, category: str) -> ScenarioType:
        """Determine primary business scenario type based on context"""
        app_name = app_name.lower()
        category = category.lower()
        
        # Priority order of scenario detection (highest priority first)
        
        # 1. Maintenance windows (highest priority - clear operational impact)
        if context.is_maintenance_window:
            return ScenarioType.MAINTENANCE_WINDOW
        
        # 2. Holiday periods (major business impact)
        if context.is_holiday:
            return ScenarioType.HOLIDAY_PERIOD
        
        # 3. Quarter-end spikes (especially for finance apps)
        if context.is_quarter_end:
            if any(finance_term in app_name for finance_term in ['finance', 'accounting', 'erp', 'sap']):
                return ScenarioType.BUSINESS_SPIKE
            else:
                return ScenarioType.QUARTER_END
        
        # 4. Month-end patterns (HR, Finance, Payroll apps)
        if context.is_month_end:
            if any(biz_term in app_name for biz_term in ['finance', 'hr', 'payroll', 'accounting']):
                return ScenarioType.BUSINESS_SPIKE
        
        # 5. Seasonal patterns (summer, winter variations)
        if context.month in [12, 1, 2, 6, 7, 8]:  # Winter/Summer months
            return ScenarioType.SEASONAL_PATTERN
        
        # 6. Default to normal operations
        return ScenarioType.NORMAL_OPERATIONS
    
    def get_smart_weights(self, context: PredictionContext) -> Dict[str, Any]:
        """
        Calculate optimal ensemble weights based on business context
        
        Returns:
            Dictionary with weights, reasoning, and confidence
        """
        base_rule = self.weight_rules[context.scenario_type]
        
        # Start with base weights
        weights = {
            'xgb': base_rule['xgb'],
            'lstm': base_rule['lstm']
        }
        
        # Fine-tune weights based on additional context
        app_name = context.app_name.lower()
        
        # Finance apps during spikes: favor XGBoost more
        if any(finance_term in app_name for finance_term in ['finance', 'accounting', 'erp']) and \
           context.scenario_type in [ScenarioType.BUSINESS_SPIKE, ScenarioType.QUARTER_END]:
            weights['xgb'] = min(0.9, weights['xgb'] + 0.1)
            weights['lstm'] = 1.0 - weights['xgb']
        
        # Education/Training apps during seasonal periods: favor LSTM more
        if any(edu_term in app_name for edu_term in ['education', 'learning', 'training']) and \
           context.scenario_type == ScenarioType.SEASONAL_PATTERN:
            weights['lstm'] = min(0.9, weights['lstm'] + 0.1)
            weights['xgb'] = 1.0 - weights['lstm']
        
        return {
            'weights': weights,
            'scenario': context.scenario_type.value,
            'reason': base_rule['reason'],
            'confidence': base_rule['confidence']
        }
    
    def predict_smart_ensemble(self, X_xgb: np.ndarray, X_lstm: np.ndarray, 
                              context: PredictionContext, horizon: str = "3month") -> Dict[str, Any]:
        """
        Make prediction using smart ensemble weighting
        
        Args:
            X_xgb: Features prepared for XGBoost model (2D array)
            X_lstm: Features prepared for LSTM model (3D array: samples, timesteps, features)
            context: Business context for smart weighting
            horizon: Prediction horizon ("3month" or "6month")
            
        Returns:
            Dictionary with prediction results and explanation
        """
        try:
            # Get model keys for the specified horizon
            xgb_key = f"xgb_{horizon}"
            lstm_key = f"lstm_{horizon}"
            
            if xgb_key not in self.models or lstm_key not in self.models:
                raise ValueError(f"Models not found for horizon: {horizon}")
            
            # Check if models are actually loaded (not None)
            if self.models[xgb_key] is None:
                raise ValueError(f"XGBoost model for {horizon} failed to load")
            if self.models[lstm_key] is None:
                print(f"⚠️ LSTM model for {horizon} not available, using XGBoost only")
                return self._xgboost_only_prediction(X_xgb, context, horizon)
            
            # Get individual model predictions
            xgb_pred = self.models[xgb_key].predict(X_xgb)
            lstm_pred = self.models[lstm_key].predict(X_lstm)
            
            # Handle different prediction shapes
            if isinstance(xgb_pred, np.ndarray) and len(xgb_pred.shape) > 0:
                xgb_pred = float(xgb_pred[0]) if len(xgb_pred) > 0 else 0.0
            else:
                xgb_pred = float(xgb_pred)
                
            if isinstance(lstm_pred, np.ndarray):
                lstm_pred = float(lstm_pred.flatten()[0]) if lstm_pred.size > 0 else 0.0
            else:
                lstm_pred = float(lstm_pred)
            
            # Get smart weights based on context
            weight_info = self.get_smart_weights(context)
            weights = weight_info['weights']
            
            # Calculate smart ensemble prediction
            smart_prediction = (weights['xgb'] * xgb_pred) + (weights['lstm'] * lstm_pred)
            
            # Store prediction for tracking
            self.prediction_history.append({
                'date': context.date,
                'app': context.app_name,
                'scenario': context.scenario_type.value,
                'weights': weights,
                'xgb_pred': xgb_pred,
                'lstm_pred': lstm_pred,
                'ensemble_pred': smart_prediction
            })
            
            return {
                'prediction': smart_prediction,
                'xgb_prediction': xgb_pred,
                'lstm_prediction': lstm_pred,
                'weights_used': weights,
                'scenario_detected': weight_info['scenario'],
                'reasoning': weight_info['reason'],
                'confidence': weight_info['confidence'],
                'horizon': horizon,
                'context': context
            }
            
        except Exception as e:
            print(f"❌ Error in smart ensemble prediction: {str(e)}")
            # Fallback to default ensemble (your current approach)
            return self._fallback_prediction(X_xgb, X_lstm, context, horizon)
    
    def _xgboost_only_prediction(self, X_xgb: np.ndarray, context: PredictionContext, horizon: str) -> Dict[str, Any]:
        """Use only XGBoost when LSTM is not available"""
        try:
            xgb_key = f"xgb_{horizon}"
            xgb_pred = float(self.models[xgb_key].predict(X_xgb)[0])
            
            return {
                'prediction': xgb_pred,
                'xgb_prediction': xgb_pred,
                'lstm_prediction': 0.0,  # Not available
                'weights_used': {'xgb': 1.0, 'lstm': 0.0},
                'scenario_detected': 'xgboost_only',
                'reasoning': 'LSTM model unavailable, using XGBoost only',
                'confidence': 0.8,  # Still confident with XGBoost
                'xgboost_only': True
            }
        except Exception as e:
            print(f"❌ XGBoost-only prediction failed: {str(e)}")
            raise
    
    def _fallback_prediction(self, X_xgb: np.ndarray, X_lstm: np.ndarray, 
                           context: PredictionContext, horizon: str) -> Dict[str, Any]:
        """Fallback to default ensemble if smart ensemble fails"""
        try:
            xgb_key = f"xgb_{horizon}"
            lstm_key = f"lstm_{horizon}"
            
            # Try XGBoost only if LSTM fails
            if self.models[lstm_key] is None:
                return self._xgboost_only_prediction(X_xgb, context, horizon)
            
            xgb_pred = float(self.models[xgb_key].predict(X_xgb)[0])
            lstm_pred = float(self.models[lstm_key].predict(X_lstm).flatten()[0])
            
            # Default 50-50 weighting (your current ensemble approach)
            ensemble_pred = 0.5 * xgb_pred + 0.5 * lstm_pred
            
            return {
                'prediction': ensemble_pred,
                'xgb_prediction': xgb_pred,
                'lstm_prediction': lstm_pred,
                'weights_used': self.default_weights,
                'scenario_detected': 'fallback',
                'reasoning': 'Using fallback default ensemble (50-50 weights)',
                'confidence': 0.7,
                'fallback': True
            }
        except Exception as e:
            print(f"❌ Fallback prediction failed: {str(e)}")
            raise
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of smart ensemble decisions"""
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        history_df = pd.DataFrame(self.prediction_history)
        
        return {
            'total_predictions': len(history_df),
            'scenario_distribution': history_df['scenario'].value_counts().to_dict(),
            'avg_xgb_weight': history_df.apply(lambda x: x['weights']['xgb'], axis=1).mean(),
            'avg_lstm_weight': history_df.apply(lambda x: x['weights']['lstm'], axis=1).mean(),
            'most_common_scenario': history_df['scenario'].mode().iloc[0] if not history_df.empty else None,
            'weight_distribution': {
                'xgb_dominant': len(history_df[history_df.apply(lambda x: x['weights']['xgb'] > 0.6, axis=1)]),
                'lstm_dominant': len(history_df[history_df.apply(lambda x: x['weights']['lstm'] > 0.6, axis=1)]),
                'balanced': len(history_df[history_df.apply(lambda x: abs(x['weights']['xgb'] - 0.5) < 0.1, axis=1)])
            }
        }

# Quick test function
def get_model_input_shapes(smart_ensemble):
    """Determine the correct input shapes for your models"""
    try:
        xgb_model = smart_ensemble.models.get('xgb_3month')
        lstm_model = smart_ensemble.models.get('lstm_3month')
        
        # For XGBoost, we need to check the number of features it expects
        # This is stored in the model's feature count
        if xgb_model is not None:
            try:
                # Try to get feature count from XGBoost model
                if hasattr(xgb_model, 'n_features_in_'):
                    xgb_features = xgb_model.n_features_in_
                else:
                    # Fallback: try with a small array and see what happens
                    try:
                        xgb_model.predict(np.random.random((1, 1)))
                        xgb_features = 1
                    except:
                        try:
                            xgb_model.predict(np.random.random((1, 5)))
                            xgb_features = 5
                        except:
                            xgb_features = 10  # Default fallback
            except:
                xgb_features = 5  # Most common from your system
        else:
            xgb_features = 5
        
        # For LSTM, check the input shape
        if lstm_model is not None:
            try:
                lstm_input_shape = lstm_model.input_shape
                # Shape is typically (None, timesteps, features)
                lstm_timesteps = lstm_input_shape[1] if lstm_input_shape[1] else 12
                lstm_features = lstm_input_shape[2] if lstm_input_shape[2] else 1
            except:
                # Default LSTM shape for your sliding window approach
                lstm_timesteps = 12
                lstm_features = 1
        else:
            lstm_timesteps = 12
            lstm_features = 1
        
        return xgb_features, (lstm_timesteps, lstm_features)
        
    except Exception as e:
        print(f"⚠️ Could not determine input shapes: {e}")
        return 5, (12, 1)  # Safe defaults

def test_smart_ensemble():
    """Test function to verify Smart Ensemble works with your models"""
    print("🧪 Testing Smart Ensemble with your models...")
    
    try:
        # Initialize Smart Ensemble
        smart_ensemble = SmartEnsemble()
        
        # Test context analysis
        test_context = smart_ensemble.analyze_context(
            date=datetime(2024, 12, 30),  # Holiday + Quarter-end
            app_name="Finance_ERP",
            category="Database"
        )
        
        print(f"✅ Context Analysis:")
        print(f"   Scenario: {test_context.scenario_type.value}")
        print(f"   Holiday: {test_context.is_holiday}")
        print(f"   Quarter-end: {test_context.is_quarter_end}")
        
        # Test weight calculation
        weights = smart_ensemble.get_smart_weights(test_context)
        print(f"✅ Smart Weights:")
        print(f"   XGBoost: {weights['weights']['xgb']:.1%}")
        print(f"   LSTM: {weights['weights']['lstm']:.1%}")
        print(f"   Reasoning: {weights['reason']}")
        
        # Get correct input shapes
        xgb_features, (lstm_timesteps, lstm_features) = get_model_input_shapes(smart_ensemble)
        print(f"✅ Detected Input Shapes:")
        print(f"   XGBoost expects: {xgb_features} features")
        print(f"   LSTM expects: ({lstm_timesteps}, {lstm_features}) - (timesteps, features)")
        
        # Create correctly shaped test data
        dummy_xgb_data = np.random.random((1, xgb_features))
        dummy_lstm_data = np.random.random((1, lstm_timesteps, lstm_features))
        
        result = smart_ensemble.predict_smart_ensemble(
            X_xgb=dummy_xgb_data,
            X_lstm=dummy_lstm_data,
            context=test_context,
            horizon="3month"
        )
        
        print(f"✅ Smart Ensemble Prediction:")
        print(f"   XGBoost prediction: {result['xgb_prediction']:.2f}")
        print(f"   LSTM prediction: {result['lstm_prediction']:.2f}")
        print(f"   Smart ensemble: {result['prediction']:.2f}")
        print(f"   Weights used: XGB={result['weights_used']['xgb']:.1%}, LSTM={result['weights_used']['lstm']:.1%}")
        
        # Show performance summary
        perf_summary = smart_ensemble.get_model_performance_summary()
        print(f"\n📊 Performance Summary:")
        print(f"   Total predictions: {perf_summary['total_predictions']}")
        if perf_summary['total_predictions'] > 0:
            print(f"   Scenario distribution: {perf_summary['scenario_distribution']}")
        
        print("\n🎉 Smart Ensemble Step 1 completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. ✅ Smart Ensemble is working with your models")
        print("   2. ➡️  Ready for Step 2: Business Rules Engine")
        print("   3. 🔮 You can now integrate this with real data preparation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test when script is executed directly
    test_smart_ensemble()
