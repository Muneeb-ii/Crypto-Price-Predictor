import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycoingecko import CoinGeckoAPI
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import seaborn as sns

# Core constants for data processing and model configuration
RESULTS_DIR = 'results/lstm_prediction_report'
HISTORICAL_DATA_DIR = 'Historical_Data'
MIN_TRAIN_SIZE = 100
MIN_TEST_SIZE = 30
TEST_PERIOD_DAYS = 180

# CoinGecko API mapping for cryptocurrency data retrieval
coin_mapping = {
    'Aave': 'aave',
    'BinanceCoin': 'binancecoin',
    'Bitcoin': 'bitcoin',
    'Cardano': 'cardano',
    'ChainLink': 'chainlink',
    'Cosmos': 'cosmos',
    'CryptocomCoin': 'crypto-com-chain',
    'Dogecoin': 'dogecoin',
    'EOS': 'eos',
    'Ethereum': 'ethereum',
    'Iota': 'iota',
    'Litecoin': 'litecoin',
    'Monero': 'monero',
    'NEM': 'nem',
    'Polkadot': 'polkadot',
    'Solana': 'solana',
    'Stellar': 'stellar',
    'Tether': 'tether',
    'Tron': 'tron',
    'USDCoin': 'usd-coin',
    'Uniswap': 'uniswap',
    'WrappedBitcoin': 'wrapped-bitcoin',
    'XRP': 'ripple'
}

# Cryptocurrency price range categorization
high_range_coins = ['bitcoin', 'wrapped-bitcoin', 'ethereum']
mid_range_coins = ['binancecoin', 'solana', 'cardano', 'polkadot', 'chainlink', 'uniswap']
price_ranges = {
    'high': high_range_coins,
    'mid': mid_range_coins,
    'low': [coin_id for coin_id in coin_mapping.values() if coin_id not in high_range_coins + mid_range_coins]
}

# LSTM model configurations optimized for different price ranges
lstm_config = {
    'high': {
        'units': 128,
        'dropout': 0.3,
        'dense_units': 64,
        'batch_size': 32,
        'epochs': 100,
        'patience': 20
    },
    'mid': {
        'units': 64,
        'dropout': 0.2,
        'dense_units': 32,
        'batch_size': 32,
        'epochs': 80,
        'patience': 15
    },
    'low': {
        'units': 32,
        'dropout': 0.1,
        'dense_units': 16,
        'batch_size': 32,
        'epochs': 60,
        'patience': 10
    }
}

# --- Feature Engineering ---
def calculate_rsi(data, periods=14):
    """Calculate RSI technical indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD technical indicator"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def create_features(df):
    """Create features for the dataset, ensuring no data leakage"""
    df = df.copy()
    
    # Calculate VWAP
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['VWAP_MA5'] = df['VWAP'].rolling(window=5, min_periods=1).mean().shift(1)
    df['VWAP_MA20'] = df['VWAP'].rolling(window=20, min_periods=1).mean().shift(1)
    df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']  # Normalized price difference from VWAP
    
    # Basic price features (no leakage as they use only current or past data)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Range_5d'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / df['Close']
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'MA{window}'] = df['Close'].rolling(window=window, min_periods=1).mean().shift(1)
        df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean().shift(1)
    
    # Volatility and momentum (ensure proper shifting)
    df['Volatility'] = df['Price_Change'].rolling(window=10, min_periods=1).std().shift(1)
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_MA5'] = df['Momentum'].rolling(5, min_periods=1).mean().shift(1)
    df['Momentum_MA20'] = df['Momentum'].rolling(20, min_periods=1).mean().shift(1)
    
    # Technical indicators (ensure proper shifting)
    df['RSI'] = calculate_rsi(df['Close']).shift(1)
    df['RSI_MA5'] = df['RSI'].rolling(5, min_periods=1).mean().shift(1)
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd.shift(1)
    df['MACD_Signal'] = signal.shift(1)
    df['MACD_Hist'] = hist.shift(1)
    
    # Bollinger Bands (ensure proper shifting)
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean().shift(1)
    df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std().shift(1)
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Price volatility over different periods
    df['Price_Volatility_5d'] = df['Price_Change'].rolling(window=5, min_periods=1).std().shift(1)
    df['Price_Volatility_20d'] = df['Price_Change'].rolling(window=20, min_periods=1).std().shift(1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def check_directories():
    """Check if required directories exist and create them if needed"""
    required_dirs = [RESULTS_DIR, HISTORICAL_DATA_DIR]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def get_price_range(coin_id):
    """Determine the price range category for a coin"""
    for range_name, coins in price_ranges.items():
        if coin_id in coins:
            return range_name
    return 'low'

def create_lstm_model(input_shape, config):
    """Create LSTM model with specified configuration"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config['units'], return_sequences=True),
        Dropout(config['dropout']),
        LSTM(config['units'] // 2, return_sequences=False),
        Dropout(config['dropout']),
        Dense(config['dense_units'], activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, window=30):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])  # Predict next day's close price
    return np.array(X), np.array(y)

def process_coin(coin, coin_id, historical_df, recent_df):
    """Process a single cryptocurrency"""
    try:
        print(f"\nProcessing {coin}...")
        
        # Debug: Print initial data shape
        print(f"Initial historical data shape: {historical_df.shape}")
        
        # Combine and validate data
        if recent_df is None:
            df = historical_df
        else:
            df = pd.concat([historical_df, recent_df])
            df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # Debug: Print combined data shape
        print(f"Combined data shape: {df.shape}")
        
        # Validate data
        if df.isnull().any().any():
            print(f"Warning: Missing values in data for {coin}, attempting to handle...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure we have enough data
        if len(df) < MIN_TRAIN_SIZE + MIN_TEST_SIZE:
            print(f"Insufficient data for {coin}, skipping...")
            return None
            
        # Split data ensuring no overlap
        split_date = df['Date'].max() - timedelta(days=TEST_PERIOD_DAYS)
        train = df[df['Date'] < split_date].copy()
        test = df[df['Date'] >= split_date].copy()
        
        # Debug: Print split data shapes
        print(f"Train data shape: {train.shape}")
        print(f"Test data shape: {test.shape}")
        
        # Verify split
        if len(train) < MIN_TRAIN_SIZE or len(test) < MIN_TEST_SIZE:
            print(f"Insufficient data after split for {coin}, skipping...")
            return None
            
        # Verify no overlap in dates
        if train['Date'].max() >= test['Date'].min():
            print(f"Warning: Date overlap detected in {coin}, adjusting split...")
            split_date = train['Date'].max() + timedelta(days=1)
            train = df[df['Date'] < split_date].copy()
            test = df[df['Date'] >= split_date].copy()
        
        # Create features
        print(f"Creating features for {coin}...")
        try:
            train = create_features(train)
            test = create_features(test)
            
            # Debug: Print feature creation results
            print(f"Train features shape: {train.shape}")
            print(f"Test features shape: {test.shape}")
            
        except ValueError as e:
            print(f"Error in feature creation for {coin}: {str(e)}")
            return None
        
        # Verify we have features
        if len(train) == 0 or len(test) == 0:
            print(f"No features created for {coin}, skipping...")
            return None
        
        # Get price range and configuration
        price_range = get_price_range(coin_id)
        config = lstm_config[price_range]
        
        # Prepare features
        feature_cols = [col for col in train.columns if col not in ['Date', 'Close']]
        
        # Debug: Print feature columns
        print(f"Number of feature columns: {len(feature_cols)}")
        
        # Scale features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        # Scale training data
        X_train_scaled = feature_scaler.fit_transform(train[feature_cols])
        y_train_scaled = target_scaler.fit_transform(train[['Close']])
        
        # Debug: Print scaled data shapes
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"y_train_scaled shape: {y_train_scaled.shape}")
        
        # Scale test data using training scalers
        X_test_scaled = feature_scaler.transform(test[feature_cols])
        y_test = test['Close'].values
        
        # Create sequences
        window = 30
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, window)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, window)
        
        # Debug: Print sequence shapes
        print(f"X_train_seq shape: {X_train_seq.shape}")
        print(f"X_test_seq shape: {X_test_seq.shape}")
        
        # Verify sequences
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Error creating sequences for {coin}, skipping...")
            return None
        
        # Create and train model
        print(f"Training LSTM model for {coin}...")
        model = create_lstm_model((window, len(feature_cols)), config)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Make predictions
        print(f"Making predictions for {coin}...")
        y_pred_scaled = model.predict(X_test_seq)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test[window:], y_pred)
        mse = mean_squared_error(y_test[window:], y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test[window:] - y_pred.flatten()) / y_test[window:])) * 100
        
        print(f"Metrics for {coin}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'coin_id': coin_id,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Price_Range': price_range,
            'test_data': test,
            'y_test': y_test[window:],
            'preds': y_pred.flatten(),
            'history': history.history
        }
        
    except Exception as e:
        print(f"Error processing {coin}: {str(e)}")
        return None

def create_plots(coin, test, y_test, preds, price_range, history, pdf):
    """Create and save plots to PDF"""
    fig = plt.figure(figsize=(15, 10))
    
    # Price Predictions
    plt.subplot(2, 2, 1)
    plt.plot(test['Date'].values[-len(y_test):], y_test, label='Actual', color='blue')
    plt.plot(test['Date'].values[-len(preds):], preds, label='Predicted', color='red')
    plt.title(f'{coin} Price Prediction (Last 6 Months) - {price_range} range')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Training History
    plt.subplot(2, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{coin} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Prediction Error Distribution
    plt.subplot(2, 2, 3)
    errors = y_test - preds
    plt.hist(errors, bins=50)
    plt.title(f'{coin} Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    # Actual vs Predicted Scatter
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title(f'{coin} Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def create_summary_plots(results_df, pdf):
    """Create summary plots for all coins"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE Comparison by Price Range
    ax = axs[0, 0]
    for range_name, color in zip(['high', 'mid', 'low'], ['tab:blue', 'tab:orange', 'tab:green']):
        range_results = results_df[results_df['Price_Range'] == range_name]
        ax.bar(range_results.index, range_results['RMSE'], label=range_name.upper(), color=color)
    ax.set_title('RMSE Comparison Across Cryptocurrencies by Price Range')
    ax.set_xlabel('Cryptocurrency')
    ax.set_ylabel('RMSE')
    ax.set_xticks(range(len(results_df.index)))
    ax.set_xticklabels(results_df.index, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    # MAPE Comparison by Price Range
    ax = axs[0, 1]
    for range_name, color in zip(['high', 'mid', 'low'], ['tab:blue', 'tab:orange', 'tab:green']):
        range_results = results_df[results_df['Price_Range'] == range_name]
        ax.bar(range_results.index, range_results['MAPE'], label=range_name.upper(), color=color)
    ax.set_title('MAPE Comparison Across Cryptocurrencies by Price Range')
    ax.set_xlabel('Cryptocurrency')
    ax.set_ylabel('MAPE (%)')
    ax.set_xticks(range(len(results_df.index)))
    ax.set_xticklabels(results_df.index, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    # Box Plot of RMSE and MAPE by Price Range
    ax = axs[1, 0]
    sns.boxplot(x='Price_Range', y='RMSE', data=results_df, ax=ax, palette='Set2')
    ax.set_title('RMSE Distribution by Price Range')
    ax.set_xlabel('Price Range')
    ax.set_ylabel('RMSE')
    ax2 = ax.twinx()
    sns.boxplot(x='Price_Range', y='MAPE', data=results_df, ax=ax2, palette='Set1', boxprops=dict(alpha=0.3))
    ax2.set_ylabel('MAPE (%)')
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels([f'{y:.1f}' for y in ax2.get_yticks()])
    ax2.grid(False)
    
    # Scatter plot of RMSE vs. MAPE
    ax = axs[1, 1]
    colors = {'high': 'tab:blue', 'mid': 'tab:orange', 'low': 'tab:green'}
    for range_name in ['high', 'mid', 'low']:
        subset = results_df[results_df['Price_Range'] == range_name]
        ax.scatter(subset['RMSE'], subset['MAPE'], label=range_name.upper(), 
                  color=colors[range_name], s=60, alpha=0.7, edgecolor='k')
    ax.set_title('RMSE vs. MAPE by Price Range')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

if __name__ == "__main__":
    # Initialize
    check_directories()
    results = {}
    
    # Process each coin
    with PdfPages(os.path.join(RESULTS_DIR, 'prediction_report.pdf')) as pdf:
        pbar = tqdm(coin_mapping.items(), total=len(coin_mapping), desc="Processing coins")
        
        for coin, coin_id in pbar:
            try:
                # Load historical data
                historical_file = os.path.join(HISTORICAL_DATA_DIR, f'coin_{coin}.csv')
                if not os.path.exists(historical_file):
                    print(f"Historical file not found for {coin}, skipping...")
                    continue
                
                historical_df = pd.read_csv(historical_file, parse_dates=['Date'])
                historical_df = historical_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Debug: Print initial data loading
                print(f"\nLoading data for {coin}...")
                print(f"Historical data shape: {historical_df.shape}")
                print(f"Date range: {historical_df['Date'].min()} to {historical_df['Date'].max()}")
                
                if historical_df.isnull().any().any():
                    print(f"Missing values in historical data for {coin}, skipping...")
                    continue
                
                # Fetch recent data
                cg = CoinGeckoAPI()
                recent_data = cg.get_coin_market_chart_by_id(
                    id=coin_id,
                    vs_currency='usd',
                    days=30
                )
                
                if recent_data and 'prices' in recent_data:
                    recent_df = pd.DataFrame(recent_data['prices'], columns=['Date', 'Close'])
                    recent_df['Date'] = pd.to_datetime(recent_df['Date'], unit='ms')
                    
                    # Adjust dates to align with historical data
                    last_historical_date = historical_df['Date'].max()
                    date_diff = recent_df['Date'].min() - last_historical_date
                    
                    if date_diff.days > 1:  # If there's a gap
                        print(f"Warning: Gap of {date_diff.days} days between historical and recent data")
                        # Adjust recent dates to start from the day after historical data
                        recent_df['Date'] = recent_df['Date'] - date_diff + timedelta(days=1)
                    
                    recent_df['Open'] = recent_df['Close']
                    recent_df['High'] = recent_df['Close']
                    recent_df['Low'] = recent_df['Close']
                    recent_df['Volume'] = 0  # Volume not available in recent data
                    
                    # Debug: Print recent data
                    print(f"Recent data shape: {recent_df.shape}")
                    print(f"Recent data date range: {recent_df['Date'].min()} to {recent_df['Date'].max()}")
                else:
                    recent_df = None
                    print("No recent data available")
                
                # Process coin
                result = process_coin(coin, coin_id, historical_df, recent_df)
                if result is None:
                    continue
                
                # Store results
                results[coin_id] = {
                    'MAE': result['MAE'],
                    'MSE': result['MSE'],
                    'RMSE': result['RMSE'],
                    'MAPE': result['MAPE'],
                    'Price_Range': result['Price_Range']
                }
                
                # Create plots
                create_plots(coin_id, result['test_data'], result['y_test'], 
                           result['preds'], result['Price_Range'], 
                           result['history'], pdf)
                
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
    print("\nPrediction Metrics Summary by Price Range:")
    for range_name in ['high', 'mid', 'low']:
        range_results = results_df[results_df['Price_Range'] == range_name]
        print(f"\n{range_name.upper()} Price Range:")
        print(range_results[['MAE', 'MSE', 'RMSE', 'MAPE']].describe())