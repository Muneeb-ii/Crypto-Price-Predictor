import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import seaborn as sns

# Constants
RESULTS_DIR = 'results/lstm_prediction_report'
HISTORICAL_DATA_DIR = 'Historical_Data'
SEQUENCE_LENGTH = 60  # Number of time steps to look back
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def check_directories():
    """Check if required directories exist and create them if needed"""
    required_dirs = [RESULTS_DIR, HISTORICAL_DATA_DIR]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def create_sequences(data, seq_length):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

@tf.function(reduce_retracing=True)
def train_step(model, optimizer, x, y):
    """Single training step with tf.function"""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.mean_squared_error(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def create_lstm_model(input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        Input(shape=input_shape, dtype=tf.float32),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        jit_compile=True  # Enable XLA compilation
    )
    return model

def create_plots(coin, test_dates, y_test, preds, pdf):
    """Create and save plots to PDF"""
    fig = plt.figure(figsize=(15, 10))
    
    # Price Predictions
    plt.subplot(2, 2, 1)
    plt.plot(test_dates, y_test, label='Actual', color='blue')
    plt.plot(test_dates, preds, label='Predicted', color='red')
    plt.title(f'{coin} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Prediction Error Distribution
    plt.subplot(2, 2, 2)
    errors = y_test - preds
    plt.hist(errors, bins=50)
    plt.title(f'{coin} Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    # Actual vs Predicted Scatter
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title(f'{coin} Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    
    # Error Over Time
    plt.subplot(2, 2, 4)
    plt.plot(test_dates, errors)
    plt.title(f'{coin} Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def create_summary_plots(results_df, pdf):
    """Create summary plots for all coins"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE Comparison
    ax = axs[0, 0]
    ax.bar(results_df.index, results_df['RMSE'])
    ax.set_title('RMSE Comparison Across Cryptocurrencies')
    ax.set_xlabel('Cryptocurrency')
    ax.set_ylabel('RMSE')
    ax.set_xticks(range(len(results_df.index)))
    ax.set_xticklabels(results_df.index, rotation=45, ha='right', fontsize=8)
    
    # MAPE Comparison
    ax = axs[0, 1]
    ax.bar(results_df.index, results_df['MAPE'])
    ax.set_title('MAPE Comparison Across Cryptocurrencies')
    ax.set_xlabel('Cryptocurrency')
    ax.set_ylabel('MAPE (%)')
    ax.set_xticks(range(len(results_df.index)))
    ax.set_xticklabels(results_df.index, rotation=45, ha='right', fontsize=8)
    
    # Box Plot of RMSE and MAPE
    ax = axs[1, 0]
    sns.boxplot(data=results_df[['RMSE', 'MAPE']], ax=ax)
    ax.set_title('RMSE and MAPE Distribution')
    ax.set_ylabel('Value')
    
    # Scatter plot of RMSE vs. MAPE
    ax = axs[1, 1]
    ax.scatter(results_df['RMSE'], results_df['MAPE'], s=60, alpha=0.7, edgecolor='k')
    ax.set_title('RMSE vs. MAPE')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('MAPE (%)')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def process_coin(coin, historical_df):
    """Process a single coin's data"""
    try:
        print(f"\nProcessing {coin}...")
        # Prepare data
        df = historical_df.copy()
        df = df[['Date', 'Close']]
        
        print(f"Data shape: {df.shape}")
        
        # Ensure minimum data length
        if len(df) < SEQUENCE_LENGTH * 3:  # Need enough data for train/val/test
            print(f"Insufficient data for {coin}, skipping...")
            return None
        
        # Split data into train (70%), validation (10%), and test (20%)
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.1)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        # Scale the data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[['Close']])
        val_scaled = scaler.transform(val_df[['Close']])
        test_scaled = scaler.transform(test_df[['Close']])
        
        # Create sequences
        print("Creating sequences...")
        X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)
        X_val, y_val = create_sequences(val_scaled, SEQUENCE_LENGTH)
        X_test, y_test = create_sequences(test_scaled, SEQUENCE_LENGTH)
        
        # Ensure sequences have correct shape
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print(f"Insufficient sequences for {coin}, skipping...")
            return None
            
        print(f"Training sequences shape: {X_train.shape}")
        
        # Create and train model
        print("Creating model...")
        model = create_lstm_model((SEQUENCE_LENGTH, 1))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join(RESULTS_DIR, f'{coin}_best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print("Making predictions...")
        # Make predictions
        test_predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, test_predictions)
        mse = mean_squared_error(y_test_actual, test_predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
        
        print(f"Metrics for {coin}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'test_dates': test_df['Date'].values[SEQUENCE_LENGTH:],
            'y_test': y_test_actual.flatten(),
            'preds': test_predictions.flatten(),
            'history': history.history
        }
        
    except Exception as e:
        print(f"Error processing {coin}: {str(e)}")
        return None

if __name__ == "__main__":
    # Initialize
    check_directories()
    results = {}
    
    # Get list of historical data files
    historical_files = [f for f in os.listdir(HISTORICAL_DATA_DIR) if f.startswith('coin_')]
    print(f"Found {len(historical_files)} historical files")
    
    # Process each coin
    with PdfPages(os.path.join(RESULTS_DIR, 'prediction_report.pdf')) as pdf:
        pbar = tqdm(historical_files, desc="Processing coins")
        
        for file in pbar:
            try:
                coin = file.replace('coin_', '').replace('.csv', '')
                print(f"\nProcessing file: {file}")
                
                # Load historical data
                historical_df = pd.read_csv(
                    os.path.join(HISTORICAL_DATA_DIR, file),
                    parse_dates=['Date']
                )
                
                if historical_df.isnull().any().any():
                    print(f"Missing values in historical data for {coin}, skipping...")
                    continue
                
                # Process coin
                result = process_coin(coin, historical_df)
                if result is None:
                    continue
                
                # Store results
                results[coin] = {
                    'MAE': result['MAE'],
                    'MSE': result['MSE'],
                    'RMSE': result['RMSE'],
                    'MAPE': result['MAPE']
                }
                
                # Create plots
                create_plots(
                    coin,
                    result['test_dates'],
                    result['y_test'],
                    result['preds'],
                    pdf
                )
                
                # Update progress bar
                pbar.set_postfix({
                    'MAPE': f"{result['MAPE']:.2f}%",
                    'RMSE': f"{result['RMSE']:.2f}"
                })
                
            except Exception as e:
                print(f"Error processing {coin}: {str(e)}")
                continue
        
        # Create summary plots
        results_df = pd.DataFrame(results).T
        create_summary_plots(results_df, pdf)
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'prediction_metrics.csv'))
    
    # Print summary
    print("\nPrediction Metrics Summary:")
    print(results_df[['MAE', 'MSE', 'RMSE', 'MAPE']].describe())
