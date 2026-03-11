import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
import joblib

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, LayerNormalization, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Machine learning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import from our established pipeline
from data_preprocessing import clean_excel_data
from feature_engineering import run_feature_engineering





def build_advanced_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """
    Build advanced LSTM model with attention mechanism and hybrid architecture
    
    Args:
        input_shape: Shape of input data (timesteps, features)
    
    Returns:
        Compiled LSTM model
    """
    print(f"\nBuilding advanced LSTM model with input shape: {input_shape}")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers for better temporal understanding
    lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
    lstm1_norm = LayerNormalization()(lstm1)
    
    # Second LSTM layer with attention
    lstm2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm1_norm)
    lstm2_norm = LayerNormalization()(lstm2)
    
    # Self-attention mechanism
    attention = Attention()([lstm2_norm, lstm2_norm])
    attention_norm = LayerNormalization()(attention)
    
    # Global average pooling to reduce dimensions
    from tensorflow.keras.layers import GlobalAveragePooling1D
    pooled = GlobalAveragePooling1D()(attention_norm)
    
    # Dense layers with residual connections
    dense1 = Dense(128, activation='relu')(pooled)
    dense1_drop = Dropout(0.3)(dense1)
    dense1_norm = LayerNormalization()(dense1_drop)
    
    dense2 = Dense(64, activation='relu')(dense1_norm)
    dense2_drop = Dropout(0.2)(dense2)
    
    # Residual connection
    from tensorflow.keras.layers import Add
    if pooled.shape[-1] == 64:
        residual = Add()([pooled, dense2_drop])
    else:
        residual_proj = Dense(64, activation='linear')(pooled)
        residual = Add()([residual_proj, dense2_drop])
    
    dense3 = Dense(32, activation='relu')(residual)
    dense3_drop = Dropout(0.1)(dense3)
    
    # Output layer
    outputs = Dense(1, activation='linear')(dense3_drop)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use adaptive learning rate and gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0, beta_1=0.9, beta_2=0.999),
        loss='huber',  # More robust to outliers
        metrics=['mae', 'mse']
    )
    
    print("\nAdvanced LSTM Model Architecture:")
    model.summary()
    
    return model


def build_sliding_window_lstm(prediction_length: int = 3) -> Model:
    """
    Build LSTM with sliding window technique for multi-step forecasting
    
    Architecture:
    - Sequence Length: 12 months input
    - Prediction Length: 3 or 6 months output
    - LSTM Hidden Units: 50 -> 25
    - Dropout Rate: 0.2
    """
    SEQUENCE_LENGTH = 12
    LSTM_HIDDEN_UNITS = 50
    LSTM_DROPOUT_RATE = 0.2
    
    print(f"\nBuilding sliding window LSTM:")
    print(f"  Input: {SEQUENCE_LENGTH} months historical data")
    print(f"  Output: {prediction_length} months forecast")
    print(f"  Architecture: {LSTM_HIDDEN_UNITS} -> {LSTM_HIDDEN_UNITS//2} units")
    
    model = Sequential([
        # First LSTM layer
        LSTM(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(25),  # hidden_units // 2 = 25
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer for multi-step prediction
        Dense(prediction_length)  # 3 or 6 outputs
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nSliding Window LSTM Architecture:")
    model.summary()
    
    return model


def build_hybrid_model(input_shape: Tuple[int, int]) -> Model:
    """
    Build hybrid model combining LSTM with CNN for better feature extraction
    """
    print(f"\nBuilding hybrid LSTM-CNN model with input shape: {input_shape}")
    
    inputs = Input(shape=input_shape)
    
    # CNN branch for local pattern detection
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv2 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(conv1)
    conv_pool = MaxPooling1D(pool_size=2)(conv2)
    
    # LSTM branch for temporal patterns
    lstm1 = LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    lstm2 = LSTM(25, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    
    # Flatten CNN output
    conv_flat = Flatten()(conv_pool)
    
    # Combine both branches
    from tensorflow.keras.layers import Concatenate
    combined = Concatenate()([lstm2, conv_flat])
    combined_norm = LayerNormalization()(combined)
    
    # Dense layers
    dense1 = Dense(64, activation='relu')(combined_norm)
    dense1_drop = Dropout(0.3)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1_drop)
    dense2_drop = Dropout(0.2)(dense2)
    
    outputs = Dense(1, activation='linear')(dense2_drop)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0008, clipnorm=1.0),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nHybrid LSTM-CNN Model Architecture:")
    model.summary()
    
    return model


def train_improved_lstm(X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Model, Dict]:
    """
    Train improved LSTM model with better training strategy
    
    Args:
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("\n=== Training Improved LSTM Model ===")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Training target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    print(f"Validation target stats: mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    
    # Build sliding window LSTM model
    prediction_length = 3  # Can be changed to 6 for 6-month forecasting
    model = build_sliding_window_lstm(prediction_length=prediction_length)
    
    # Learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 30:
            return lr * 0.95
        else:
            return lr * 0.9
    
    # User-specified callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # User-specified patience
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/lstm_model_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("\nTraining enhanced LSTM model...")
    print("Enhanced architecture with advanced preprocessing")
    print("Batch size: 32, Epochs: 100, Patience: 10")
    
    # Train enhanced model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Get initial performance
    val_loss = min(history.history['val_loss'])
    print(f"\nEnhanced LSTM validation loss: {val_loss:.4f}")
    
    # Try hybrid model if enhanced model doesn't perform well
    if val_loss > 0.5:  # If validation loss is still high
        print("\nTrying hybrid LSTM-CNN model for better performance...")
        
        hybrid_model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
        
        hybrid_history = hybrid_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        hybrid_val_loss = min(hybrid_history.history['val_loss'])
        print(f"\nHybrid model validation loss: {hybrid_val_loss:.4f}")
        
        # Use the better model
        if hybrid_val_loss < val_loss:
            print("\n✅ Using hybrid model (better performance)")
            model = hybrid_model
            history = hybrid_history.history
        else:
            print("\n✅ Using enhanced LSTM model (better performance)")
            history = history.history
    else:
        print("\n✅ Enhanced LSTM model performing well")
        history = history.history
    
    return model, history


def calculate_lstm_metrics(model: Model, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calculate metrics for LSTM model
    
    Args:
        model: Trained LSTM model
        X: Input sequences
        y: True targets
    
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X, verbose=0)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    def calculate_mape(y_true, y_pred):
        # Avoid division by zero
        mask = y_true != 0
        if not any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'mape': calculate_mape(y, y_pred)
    }
    
    return metrics


def calculate_lstm_metrics_scaled(model: Model, X: np.ndarray, y: np.ndarray, scalers: Dict) -> Dict:
    """
    Calculate metrics for LSTM model with inverse scaling and NaN handling
    
    Args:
        model: Trained LSTM model
        X: Input sequences (scaled)
        y: True targets (scaled)
        scalers: Dictionary containing scalers
    
    Returns:
        Dictionary of metrics on original scale
    """
    try:
        # Make predictions on scaled data
        y_pred_scaled = model.predict(X, verbose=0)
        
        # Handle NaN values in predictions
        y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Inverse transform to original scale
        target_scaler = scalers['target_scaler']
        y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Handle NaN values after inverse transform
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        def calculate_mape(y_true, y_pred):
            # Avoid division by zero
            mask = y_true != 0
            if not any(mask):
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Calculate metrics on original scale
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': calculate_mape(y_true, y_pred)
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error in metrics calculation: {str(e)}")
        # Return default metrics if calculation fails
        return {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': -1.0
        }


def create_sliding_window_data(df: pd.DataFrame, sequence_length: int = 12, prediction_length: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create sliding window data for time series forecasting
    
    Args:
        df: Preprocessed DataFrame with month columns
        sequence_length: Input sequence length (12 months)
        prediction_length: Output prediction length (3 or 6 months)
    
    Returns:
        X_train, y_train, X_val, y_val, scalers
    """
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    print(f"\nCreating sliding window data:")
    print(f"  Sequence length: {sequence_length} months")
    print(f"  Prediction length: {prediction_length} months")
    
    # Get month columns (assuming they are in YYYY-MM-DD format)
    month_cols = [col for col in df.columns if isinstance(col, str) and col.count('-') == 2 and len(col) == 10]
    month_cols = sorted(month_cols)  # Ensure chronological order
    
    print(f"  Found {len(month_cols)} month columns")
    print(f"  Date range: {month_cols[0]} to {month_cols[-1]}")
    
    if len(month_cols) < sequence_length + prediction_length:
        raise ValueError(f"Not enough months for sliding window: need {sequence_length + prediction_length}, got {len(month_cols)}")
    
    X_sequences = []
    y_sequences = []
    
    # Process each app-category group
    for key, group in df.groupby(['App Name', 'AIMS Tkt Category']):
        if len(group) == 0:
            continue
        # Extract app and category from the key tuple
        if isinstance(key, tuple):
            app, category = key
        else:
            app = category = key
            
        # Get time series values for this group
        values = group[month_cols].values[0]  # Take first row (should be same for all rows in group)
        
        # Handle NaN values
        values = np.nan_to_num(values, nan=0.0)
        
        # Create sliding windows
        for i in range(len(values) - sequence_length - prediction_length + 1):
            # Input sequence (12 months)
            X_seq = values[i:i + sequence_length].reshape(-1, 1)  # Shape: (12, 1)
            
            # Target sequence (3 or 6 months)
            y_seq = values[i + sequence_length:i + sequence_length + prediction_length]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
    
    if len(X_sequences) == 0:
        raise ValueError("No valid sequences created!")
    
    X = np.array(X_sequences)  # Shape: (n_samples, 12, 1)
    y = np.array(y_sequences)  # Shape: (n_samples, prediction_length)
    
    print(f"  Created {len(X)} sequences")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    
    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    
    # Scale targets
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    
    scalers = {'scaler': scaler}
    
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Validation samples: {len(X_val_scaled)}")
    print(f"  Data range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scalers


def train_sliding_window_lstm(X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray, 
                              prediction_length: int) -> Tuple[Model, Dict]:
    """
    Train sliding window LSTM model
    """
    print(f"\n=== Training Sliding Window LSTM ===")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Build model
    model = build_sliding_window_lstm(prediction_length=prediction_length)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/sliding_window_lstm_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("\nTraining sliding window LSTM...")
    print("Architecture: 50 -> 25 units, multi-step forecasting")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    return model, history.history


def calculate_multistep_metrics(model: Model, X: np.ndarray, y: np.ndarray, scalers: Dict) -> Dict:
    """
    Calculate metrics for multi-step forecasting
    """
    try:
        # Make predictions
        y_pred_scaled = model.predict(X, verbose=0)
        
        # Inverse transform to original scale
        scaler = scalers['scaler']
        y_true = scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        
        # Handle NaN values
        y_true = np.nan_to_num(y_true, nan=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        def calculate_mape(y_true, y_pred):
            # Avoid division by zero
            mask = y_true != 0
            if not any(mask):
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Calculate metrics for each prediction step
        step_metrics = {}
        for step in range(y_true.shape[1]):
            step_mse = mean_squared_error(y_true[:, step], y_pred[:, step])
            step_mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
            step_r2 = r2_score(y_true[:, step], y_pred[:, step])
            step_mape = calculate_mape(y_true[:, step], y_pred[:, step])
            
            step_metrics[f'step_{step+1}'] = {
                'mse': step_mse,
                'rmse': np.sqrt(step_mse),
                'mae': step_mae,
                'r2': step_r2,
                'mape': step_mape
            }
        
        # Overall metrics (average across all steps)
        overall_mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        overall_metrics = {
            'mse': overall_mse,
            'rmse': np.sqrt(overall_mse),
            'mae': mean_absolute_error(y_true.flatten(), y_pred.flatten()),
            'r2': r2_score(y_true.flatten(), y_pred.flatten()),
            'mape': calculate_mape(y_true.flatten(), y_pred.flatten())
        }
        
        # Combine step-wise and overall metrics
        metrics = {'overall': overall_metrics, 'steps': step_metrics}
        
        return metrics
        
    except Exception as e:
        print(f"Error in multistep metrics calculation: {str(e)}")
        return {
            'overall': {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2': -1.0},
            'steps': {}
        }


def run_sliding_window_lstm_training(df: pd.DataFrame, scalers: Dict, prediction_length: int = 3) -> Tuple[Optional[Model], Optional[Dict], Optional[Dict]]:
    """
    Run sliding window LSTM training pipeline
    
    Args:
        df: Preprocessed DataFrame from data_preprocessing
        scalers: Scalers dictionary from data_preprocessing
        prediction_length: Number of months to forecast (3 or 6)
    
    Returns:
        Tuple of (trained_model, train_metrics, val_metrics)
    """
    try:
        print("\n=== Sliding Window LSTM Training Pipeline ===")
        
        # Step 1: Create sliding window data directly from preprocessed DataFrame
        X_train, y_train, X_val, y_val, lstm_scalers = create_sliding_window_data(
            df, sequence_length=12, prediction_length=prediction_length
        )
        
        # Step 2: Train Sliding Window LSTM Model
        model, history = train_sliding_window_lstm(X_train, y_train, X_val, y_val, prediction_length)
        
        # Step 3: Calculate metrics for multi-step forecasting
        print("\nCalculating multi-step forecasting metrics...")
        train_metrics = calculate_multistep_metrics(model, X_train, y_train, lstm_scalers)
        val_metrics = calculate_multistep_metrics(model, X_val, y_val, lstm_scalers)
        
        # Step 4: Display Results
        print("\n=== Sliding Window LSTM Training Results ===")
        
        print("\nTraining Metrics (Overall):")
        for metric, value in train_metrics['overall'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nValidation Metrics (Overall):")
        for metric, value in val_metrics['overall'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Display step-wise metrics
        print("\nStep-wise Validation Metrics:")
        for step, metrics in val_metrics['steps'].items():
            print(f"  {step}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAPE = {metrics['mape']:.2f}%")
        
        # Step 5: Save model and metrics
        model_path = f'models/sliding_window_lstm_{prediction_length}month.h5'
        model.save(model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save metrics
        if not os.path.exists('metrics'):
            os.makedirs('metrics')
        
        metrics_path = f'metrics/lstm_metrics_{prediction_length}month.joblib'
        joblib.dump(val_metrics['overall'], metrics_path)
        print(f"✅ Metrics saved to: {metrics_path}")
        
        return model, train_metrics, val_metrics
        
    except Exception as e:
        print(f"\nError in LSTM training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    try:
        # Step 1: Data Preprocessing
        file_path = "App wise actuals_2023 to 2024.xlsx"
        print("\n=== LSTM Model Training ===")
        print(f"Processing file: {file_path}")
        
        df, scalers = clean_excel_data(file_path)
        
        if df is None:
            raise ValueError("Data preprocessing failed!")
        
        # Run Sliding Window LSTM training
        prediction_length = 3  # Change to 6 for 6-month forecasting
        model, train_metrics, val_metrics = run_sliding_window_lstm_training(df, scalers, prediction_length)
        
        if model is not None:
            print("\n🎉 LSTM training completed successfully!")
        else:
            print("\n❌ LSTM training failed!")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

