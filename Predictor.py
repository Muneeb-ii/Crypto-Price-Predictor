import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import time


# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

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

def fetch_coingecko_data(coin_id, days=365):
    """Fetch historical data from CoinGecko API"""
    try:
        cg = CoinGeckoAPI()
        data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )
        
        # Process prices data
        prices = np.array(data['prices'])
        volumes = np.array(data['total_volumes'])
        
        # Convert timestamps to dates and create DataFrame
        df = pd.DataFrame({
            'Date': [datetime.fromtimestamp(ts/1000) for ts in prices[:, 0]],
            'Close': prices[:, 1],
            'Volume': volumes[:, 1]
        })
        
        # Add OHLC data (using Close as approximation since API doesn't provide OHLC)
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close']
        df['Low'] = df['Close']
        
        # Drop first row (NaN Open)
        df = df.dropna()
        
        return df
    
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {str(e)}")
        return None

# CoinGecko ID mapping
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

# Define high and mid price range coin lists first
high_range_coins = ['bitcoin', 'wrapped-bitcoin', 'ethereum']
mid_range_coins = ['binancecoin', 'solana', 'cardano', 'polkadot', 'chainlink', 'uniswap']

# Now define price_ranges using those lists
price_ranges = {
    'high': high_range_coins,
    'mid': mid_range_coins,
    'low': [coin_id for coin_id in coin_mapping.values() if coin_id not in high_range_coins + mid_range_coins]
}

# Define range-specific features and parameters
range_config = {
    'high': {
        'features': [
            'Prev_Close', 'Open', 'High', 'Low', 'Volume',
            'MA5', 'MA10', 'MA20', 'MA50',  # More moving averages for high-value coins
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d',
            'Volatility', 'High_Low_Range', 'Price_Range_5d',
            'Momentum', 'Momentum_MA5', 'Momentum_MA20',
            'RSI', 'RSI_MA5',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volume_MA5', 'Volume_MA20',  # Volume moving averages
            'Price_Volatility_5d', 'Price_Volatility_20d'  # Price volatility over different periods
        ],
        'model_params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    },
    'mid': {
        'features': [
            'Prev_Close', 'Open', 'High', 'Low', 'Volume',
            'MA5', 'MA10', 'MA20',
            'Price_Change', 'Price_Change_5d',
            'Volatility', 'High_Low_Range', 'Price_Range_5d',
            'Momentum', 'Momentum_MA5',
            'RSI',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position'
        ],
        'model_params': {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4
        }
    },
    'low': {
        'features': [
            'Prev_Close', 'Open', 'High', 'Low', 'Volume',
            'MA5', 'MA10',
            'Price_Change', 'Price_Change_5d',
            'Volatility', 'High_Low_Range',
            'Momentum',
            'RSI',
            'MACD', 'MACD_Signal',
            'BB_Middle', 'BB_Upper', 'BB_Lower'
        ],
        'model_params': {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 15,
            'min_samples_leaf': 5
        }
    }
}

# Dictionary to store results for each coin
results = {}
feature_importance = {}

def get_price_range(coin_id):
    """Determine the price range category for a coin"""
    for range_name, coins in price_ranges.items():
        if coin_id in coins:
            return range_name
    return 'low'  # default to low range if not found

def create_range_specific_features(df):
    """Create additional features specific to each price range"""
    # High-value specific features
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean().shift(1)
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean().shift(1)
    df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
    df['Momentum_MA20'] = df['Close'] - df['Close'].shift(20)
    df['RSI_MA5'] = df['RSI'].rolling(5, min_periods=1).mean()
    df['Volume_MA5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Price_Volatility_5d'] = df['Price_Change'].rolling(5, min_periods=1).std()
    df['Price_Volatility_20d'] = df['Price_Change'].rolling(20, min_periods=1).std()
    
    return df

# Process each coin
for coin, coin_id in coin_mapping.items():
    print(f"\nProcessing {coin}...")
    
    # Read historical data
    historical_file = f'Historical_Data/coin_{coin}.csv'
    if os.path.exists(historical_file):
        historical_df = pd.read_csv(historical_file, parse_dates=['Date'])
        historical_df = historical_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Historical data shape: {historical_df.shape}")
    else:
        print(f"Historical file not found for {coin}, skipping...")
        continue
    
    # Fetch recent data from CoinGecko
    recent_df = fetch_coingecko_data(coin_id)
    if recent_df is None:
        print(f"Failed to fetch recent data for {coin}, using only historical data...")
        df = historical_df
    else:
        print(f"Recent data shape: {recent_df.shape}")
        df = pd.concat([historical_df, recent_df])
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        print(f"Combined data shape: {df.shape}")
    
    # Create base features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean().shift(1)
    df['MA10'] = df['Close'].rolling(10, min_periods=1).mean().shift(1)
    df['Volatility'] = df['Price_Change'].rolling(window=10, min_periods=1).std()
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Range_5d'] = (df['High'].rolling(5, min_periods=1).max() - df['Low'].rolling(5, min_periods=1).min()) / df['Close']
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_MA5'] = df['Momentum'].rolling(5, min_periods=1).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Create range-specific features
    df = create_range_specific_features(df)
    
    # Drop rows with NaN values after all feature engineering
    print("\nNaN counts before dropna:")
    print(df.isna().sum())
    df.dropna(inplace=True)
    print(f"Data shape after feature creation and dropna: {df.shape}")

    if len(df) == 0:
        print("No data left after feature creation, skipping...")
        continue
    
    # Get price range and corresponding configuration
    price_range = get_price_range(coin_id)
    config = range_config[price_range]
    
    # Split data
    split_date = df['Date'].max() - timedelta(days=180)
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    
    print(f"Split date: {split_date}")
    print(f"Train set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Using {price_range} range configuration")
    
    if len(train) == 0 or len(test) == 0:
        print(f"Skipping {coin} - insufficient data")
        continue
    
    # Initialize scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Scale features and target
    X_train = feature_scaler.fit_transform(train[config['features']])
    y_train = target_scaler.fit_transform(train[['Close']]).ravel()
    
    # Feature Selection using RFE
    print("\nPerforming RFE feature selection...")
    rfe = RFE(
        estimator=RandomForestRegressor(**config['model_params']),
        n_features_to_select=min(10, len(config['features'])),
        step=1
    )
    rfe.fit(X_train, y_train)
    
    # Feature Selection using LASSO
    print("Performing LASSO feature selection...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    
    # Get selected features
    rfe_features = [f for f, s in zip(config['features'], rfe.support_) if s]
    lasso_features = [f for f, c in zip(config['features'], lasso.coef_) if abs(c) > 0]
    selected_features = list(set(rfe_features + lasso_features))
    
    print(f"\nSelected features for {coin}:")
    print("RFE features:", rfe_features)
    print("LASSO features:", lasso_features)
    print("Combined features:", selected_features)
    
    # Use only selected features
    X_train_selected = X_train[:, [config['features'].index(f) for f in selected_features]]
    X_test = feature_scaler.transform(test[config['features']])
    X_test_selected = X_test[:, [config['features'].index(f) for f in selected_features]]
    y_test = test['Close'].values
    
    # Train model with selected features and range-specific parameters
    model = RandomForestRegressor(**config['model_params'])
    model.fit(X_train_selected, y_train)
    
    # Store feature importance
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance[coin] = importance
    
    # Make predictions
    preds_scaled = model.predict(X_test_selected)
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    # Store results
    results[coin] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Selected_Features': selected_features,
        'Price_Range': price_range
    }
    
    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(test['Date'], y_test, label='Actual')
    plt.plot(test['Date'], preds, label='Predicted')
    plt.title(f'{coin} Price Prediction (Last 6 Months) - {price_range} range')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'predictions_{coin}.png'))
    plt.close()
    
    time.sleep(2)

# Create a summary DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.round(2)

# Save results to CSV
results_df.to_csv(os.path.join(results_dir, 'prediction_metrics.csv'))

# Print summary by price range
print("\nPrediction Metrics Summary by Price Range:")
for range_name in ['high', 'mid', 'low']:
    range_results = results_df[results_df['Price_Range'] == range_name]
    print(f"\n{range_name.upper()} Price Range:")
    print(range_results[['MAE', 'MSE', 'RMSE', 'MAPE']].describe())

# Plot comparison of RMSE across coins by price range
plt.figure(figsize=(12,6))
for range_name in ['high', 'mid', 'low']:
    range_results = results_df[results_df['Price_Range'] == range_name]
    plt.bar(range_results.index, range_results['RMSE'], label=range_name.upper())
plt.title('RMSE Comparison Across Cryptocurrencies by Price Range')
plt.xlabel('Cryptocurrency')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'rmse_comparison_by_range.png'))
plt.close()