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
import seaborn as sns

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

# Dictionary to store results for each coin
results = {}
feature_importance = {}

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
    
    # Add RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Add MACD
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    # Add Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']  # Normalized width
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])  # Position within bands
    
    # Add Ichimoku Cloud
    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
    high_9 = df['High'].rolling(window=9, min_periods=1).max()
    low_9 = df['Low'].rolling(window=9, min_periods=1).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
    high_26 = df['High'].rolling(window=26, min_periods=1).max()
    low_26 = df['Low'].rolling(window=26, min_periods=1).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods
    high_52 = df['High'].rolling(window=52, min_periods=1).max()
    low_52 = df['Low'].rolling(window=52, min_periods=1).min()
    df['Senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Current closing price, plotted 26 periods behind
    df['Chikou_span'] = df['Close'].shift(-26)
    
    # Additional Ichimoku-derived features
    df['Cloud_Color'] = np.where(df['Senkou_span_a'] > df['Senkou_span_b'], 1, -1)  # 1 for bullish, -1 for bearish
    df['Price_vs_Cloud'] = np.where(df['Close'] > df['Senkou_span_a'], 1, -1)  # 1 if price above cloud, -1 if below
    df['TK_Cross'] = np.where(df['Tenkan_sen'] > df['Kijun_sen'], 1, -1)  # 1 for bullish cross, -1 for bearish cross
    
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
    all_features = [
        'Prev_Close', 'Open', 'High', 'Low', 'Volume', 
        'MA5', 'MA10', 'Price_Change', 'Price_Change_5d',
        'Volatility', 'High_Low_Range', 'Price_Range_5d',
        'Momentum', 'Momentum_MA5', 'RSI',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position'
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
    X_train = feature_scaler.fit_transform(train[all_features])
    y_train = target_scaler.fit_transform(train[['Close']]).ravel()
    
    # Feature Selection using RFE
    print("\nPerforming RFE feature selection...")
    rfe = RFE(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        n_features_to_select=10,  # Select top 10 features
        step=1
    )
    rfe.fit(X_train, y_train)
    
    # Feature Selection using LASSO
    print("Performing LASSO feature selection...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    
    # Get selected features from both methods
    rfe_features = [f for f, s in zip(all_features, rfe.support_) if s]
    lasso_features = [f for f, c in zip(all_features, lasso.coef_) if abs(c) > 0]
    
    # Combine features from both methods (union)
    selected_features = list(set(rfe_features + lasso_features))
    print(f"\nSelected features for {coin}:")
    print("RFE features:", rfe_features)
    print("LASSO features:", lasso_features)
    print("Combined features:", selected_features)
    
    # Use only selected features
    X_train_selected = X_train[:, [all_features.index(f) for f in selected_features]]
    X_test = feature_scaler.transform(test[all_features])
    X_test_selected = X_test[:, [all_features.index(f) for f in selected_features]]
    y_test = test['Close'].values
    
    # Train model with selected features
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_selected, y_train)
    
    # Store feature importance
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance[coin] = importance
    
    # Make predictions and inverse transform
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
        'Selected_Features': selected_features
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
    plt.savefig(os.path.join(results_dir, f'predictions_{coin}.png'))
    plt.close()
    
    # Add a small delay between API calls
    time.sleep(2)

# Create a summary DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.round(2)

# Save results to CSV
results_df.to_csv(os.path.join(results_dir, 'prediction_metrics.csv'))

# Print summary
print("\nPrediction Metrics Summary:")
print(results_df)

# Analyze feature importance by price range
def get_price_range(coin):
    avg_price = historical_df[historical_df['Date'] >= split_date]['Close'].mean()
    if avg_price > 1000:
        return 'High'
    elif avg_price > 10:
        return 'Medium'
    else:
        return 'Low'

# Group coins by price range
price_ranges = {coin: get_price_range(coin) for coin in results.keys()}
range_groups = {'High': [], 'Medium': [], 'Low': []}
for coin, price_range in price_ranges.items():
    range_groups[price_range].append(coin)

# Calculate average feature importance for each price range
range_importance = {}
for price_range, coins in range_groups.items():
    if coins:
        # Combine feature importance for all coins in this range
        combined_importance = pd.concat([feature_importance[coin] for coin in coins])
        # Calculate mean importance for each feature
        mean_importance = combined_importance.groupby('Feature')['Importance'].mean().reset_index()
        mean_importance = mean_importance.sort_values('Importance', ascending=False)
        range_importance[price_range] = mean_importance

# Plot feature importance for each price range
plt.figure(figsize=(15, 10))
for i, (price_range, importance) in enumerate(range_importance.items(), 1):
    plt.subplot(3, 1, i)
    sns.barplot(data=importance.head(10), x='Importance', y='Feature')
    plt.title(f'Top 10 Most Important Features - {price_range} Price Range')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_importance_by_range.png'))
plt.close()

# Save detailed feature importance to Excel
with pd.ExcelWriter(os.path.join(results_dir, 'feature_importance_analysis.xlsx')) as writer:
    # Save overall results
    results_df.to_excel(writer, sheet_name='Prediction Metrics')
    
    # Save feature importance for each price range
    for price_range, importance in range_importance.items():
        importance.to_excel(writer, sheet_name=f'{price_range} Range Features', index=False)
    
    # Save individual coin feature importance
    for coin, importance in feature_importance.items():
        importance.to_excel(writer, sheet_name=f'{coin} Features', index=False)

# Print top features for each price range
print("\nTop 5 Most Important Features by Price Range:")
for price_range, importance in range_importance.items():
    print(f"\n{price_range} Price Range:")
    print(importance.head().to_string(index=False))

# Plot comparison of RMSE across coins
plt.figure(figsize=(12,6))
results_df['RMSE'].sort_values().plot(kind='bar')
plt.title('RMSE Comparison Across Cryptocurrencies')
plt.xlabel('Cryptocurrency')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'rmse_comparison.png'))
plt.close()