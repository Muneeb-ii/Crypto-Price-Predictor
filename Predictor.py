import pandas as pd
import glob, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import time

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

# Dictionary to store results for each coin
results = {}

# Process each coin
for coin, coin_id in coin_mapping.items():
    print(f"\nProcessing {coin}...")
    
    # Read historical data
    historical_file = f'Historical_Data/coin_{coin}.csv'
    if os.path.exists(historical_file):
        historical_df = pd.read_csv(historical_file, parse_dates=['Date'])
        # Keep only necessary columns
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
        # Combine historical and recent data
        df = pd.concat([historical_df, recent_df])
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        print(f"Combined data shape: {df.shape}")
    
    # Create features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean().shift(1)
    df['MA10'] = df['Close'].rolling(10, min_periods=1).mean().shift(1)
    
    # Add volatility and momentum features
    df['Volatility'] = df['Price_Change'].rolling(window=10, min_periods=1).std()
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Range_5d'] = (df['High'].rolling(5, min_periods=1).max() - df['Low'].rolling(5, min_periods=1).min()) / df['Close']
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_MA5'] = df['Momentum'].rolling(5, min_periods=1).mean()
    
    # Print NaN counts for each feature
    print("\nNaN counts before dropna:")
    print(df.isna().sum())
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    print(f"\nData shape after feature creation: {df.shape}")
    
    if len(df) == 0:
        print("No data left after feature creation, skipping...")
        continue
    
    # Define features and target
    features = [
        'Prev_Close', 'Open', 'High', 'Low', 'Volume', 
        'MA5', 'MA10', 'Price_Change', 'Price_Change_5d',
        'Volatility', 'High_Low_Range', 'Price_Range_5d',
        'Momentum', 'Momentum_MA5'
    ]
    
    # Split data - use last 6 months as test set
    split_date = df['Date'].max() - timedelta(days=180)
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    
    print(f"Split date: {split_date}")
    print(f"Train set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    
    if len(train) == 0 or len(test) == 0:
        print(f"Skipping {coin} - insufficient data")
        continue
    
    # Initialize scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Scale features and target
    X_train = feature_scaler.fit_transform(train[features])
    y_train = target_scaler.fit_transform(train[['Close']]).ravel()
    
    X_test = feature_scaler.transform(test[features])
    y_test = test['Close'].values
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions and inverse transform
    preds_scaled = model.predict(X_test)
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
        'MAPE': mape
    }
    
    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(test['Date'], y_test, label='Actual')
    plt.plot(test['Date'], preds, label='Predicted')
    plt.title(f'{coin} Price Prediction (Last 6 Months)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predictions_{coin}.png')
    plt.close()
    
    # Add a small delay between API calls (2 seconds is more than enough for 30 calls/minute)
    time.sleep(2)

# Create a summary DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.round(2)

# Save results to CSV
results_df.to_csv('prediction_metrics.csv')

# Print summary
print("\nPrediction Metrics Summary:")
print(results_df)

# Plot comparison of RMSE across coins
plt.figure(figsize=(12,6))
results_df['RMSE'].sort_values().plot(kind='bar')
plt.title('RMSE Comparison Across Cryptocurrencies')
plt.xlabel('Cryptocurrency')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rmse_comparison.png')
plt.close()