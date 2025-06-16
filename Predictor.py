import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import time
from xgboost import XGBRegressor
import sys
import gc
import psutil
from tqdm import tqdm

def check_directories():
    """Check if required directories exist and create them if needed"""
    required_dirs = ['results', 'Historical_Data']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Check directories before starting
    check_directories()
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def clear_memory():
    """Clear memory by running garbage collection"""
    gc.collect()

def check_memory_threshold(threshold_mb=1000):
    """Check if memory usage is above threshold"""
    memory_usage = get_memory_usage()
    if memory_usage > threshold_mb:
        print(f"Warning: High memory usage ({memory_usage:.2f} MB)")
        clear_memory()
        return True
    return False

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
    """Fetch historical data from CoinGecko API with improved error handling"""
    try:
        cg = CoinGeckoAPI()
        data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )
        
        if not data or 'prices' not in data or not data['prices']:
            print(f"No data received for {coin_id}")
            return None
        
        # Process prices data
        prices = np.array(data['prices'])
        volumes = np.array(data['total_volumes'])
        
        # Validate data
        if len(prices) == 0 or len(volumes) == 0:
            print(f"Empty data arrays for {coin_id}")
            return None
        
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
        
        # Validate data quality
        if df['Close'].isnull().any() or df['Volume'].isnull().any():
            print(f"Missing values in data for {coin_id}")
            return None
        
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

def create_ensemble_model(price_range):
    """Create an ensemble model based on price range with improved parameters"""
    if price_range == 'high':
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8  # Add subsampling for better generalization
            )),
            ('xgb', XGBRegressor(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8  # Add column sampling
            ))
        ]
    elif price_range == 'mid':
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                subsample=0.8
            )),
            ('xgb', XGBRegressor(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8
            ))
        ]
    else:  # low price range
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.15,
                random_state=42,
                subsample=0.8
            )),
            ('xgb', XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.15,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8
            ))
        ]
    
    return VotingRegressor(estimators=models, weights=[1, 1, 1])

def create_features(df, is_training=True):
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

def create_plots(coin, test, y_test, preds, price_range, feature_importance, pdf):
    """Create and save plots to PDF"""
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Price Predictions
    plt.subplot(2, 2, 1)
    plt.plot(test['Date'], y_test, label='Actual', color='blue')
    plt.plot(test['Date'], preds, label='Predicted', color='red')
    plt.title(f'{coin} Price Prediction (Last 6 Months) - {price_range} range')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Feature Importance
    plt.subplot(2, 2, 2)
    importance = feature_importance[coin]
    plt.barh(importance['Feature'], importance['Importance'])
    plt.title(f'{coin} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # Plot 3: Prediction Error Distribution
    plt.subplot(2, 2, 3)
    errors = y_test - preds
    plt.hist(errors, bins=50)
    plt.title(f'{coin} Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    # Plot 4: Actual vs Predicted Scatter
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
    # Plot 1: RMSE Comparison by Price Range
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for range_name in ['high', 'mid', 'low']:
        range_results = results_df[results_df['Price_Range'] == range_name]
        plt.bar(range_results.index, range_results['RMSE'], label=range_name.upper())
    plt.title('RMSE Comparison Across Cryptocurrencies by Price Range')
    plt.xlabel('Cryptocurrency')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 2: MAPE Comparison by Price Range
    plt.subplot(2, 2, 2)
    for range_name in ['high', 'mid', 'low']:
        range_results = results_df[results_df['Price_Range'] == range_name]
        plt.bar(range_results.index, range_results['MAPE'], label=range_name.upper())
    plt.title('MAPE Comparison Across Cryptocurrencies by Price Range')
    plt.xlabel('Cryptocurrency')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 3: Box Plot of Metrics by Price Range
    plt.subplot(2, 2, 3)
    metrics = ['RMSE', 'MAPE']
    data = []
    labels = []
    for range_name in ['high', 'mid', 'low']:
        range_results = results_df[results_df['Price_Range'] == range_name]
        for metric in metrics:
            data.append(range_results[metric])
            labels.append(f'{range_name.upper()}\n{metric}')
    plt.boxplot(data, labels=labels)
    plt.title('Distribution of Metrics by Price Range')
    plt.xticks(rotation=45)
    
    # Plot 4: Average Metrics by Price Range
    plt.subplot(2, 2, 4)
    avg_metrics = results_df.groupby('Price_Range')[['RMSE', 'MAPE']].mean()
    avg_metrics.plot(kind='bar')
    plt.title('Average Metrics by Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Value')
    plt.legend(['RMSE', 'MAPE'])
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def process_coin(coin_id, historical_data, recent_data, pdf):
    """Process a single cryptocurrency"""
    try:
        print(f"\nProcessing {coin_id}...")
        print(f"Current memory usage: {get_memory_usage():.2f} MB")
        
        # Combine historical and recent data
        combined_data = pd.concat([historical_data, recent_data])
        print(f"Historical data shape: {historical_data.shape}")
        print(f"Recent data shape: {recent_data.shape}")
        print(f"Combined data shape: {combined_data.shape}")
        
        # Create features
        print("\nCreating features...")
        combined_data = create_features(combined_data)
        
        # Verify features were created
        required_features = [
            'Price_Change_20d', 'Price_Range_5d', 'Momentum_MA20', 
            'RSI_MA5', 'Price_Volatility_5d', 'Price_Volatility_20d'
        ]
        missing_features = [f for f in required_features if f not in combined_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Split data into training and testing sets
        train_data = combined_data[combined_data.index < recent_data.index[0]]
        test_data = combined_data[combined_data.index >= recent_data.index[0]]
        
        # Prepare features and target
        feature_columns = [col for col in combined_data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        X_train = train_data[feature_columns]
        y_train = train_data['Close']
        X_test = test_data[feature_columns]
        y_test = test_data['Close']
        
        # Perform feature selection
        print("\nPerforming RFE feature selection...")
        rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10)
        rfe.fit(X_train, y_train)
        selected_features_rfe = X_train.columns[rfe.support_].tolist()
        
        print("Performing LASSO feature selection...")
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train, y_train)
        selected_features_lasso = X_train.columns[lasso.coef_ != 0].tolist()
        
        # Combine selected features
        selected_features = list(set(selected_features_rfe + selected_features_lasso))
        if not selected_features:
            selected_features = feature_columns[:10]  # Fallback to first 10 features if none selected
        
        print(f"Selected features: {selected_features}")
        
        # Update feature sets with selected features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # Train model
        model = create_ensemble_model(get_price_range(coin_id))
        model.fit(X_train, y_train)
        
        # Store feature importance (using RandomForest's importance as reference)
        rf_model = RandomForestRegressor(**range_config[get_price_range(coin_id)]['model_params'])
        rf_model.fit(X_train, y_train)
        importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        feature_importance[coin_id] = importance
        
        # Make predictions
        preds_scaled = model.predict(X_test)
        preds = preds_scaled
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        # Create plots
        create_plots(coin_id, test_data, y_test, preds, get_price_range(coin_id), feature_importance, pdf)
        
        return {
            'coin_id': coin_id,
            'train_mape': mape,
            'test_mape': mape,
            'train_rmse': rmse,
            'test_rmse': rmse,
            'selected_features': selected_features
        }
        
    except Exception as e:
        print(f"Error processing {coin_id}: {str(e)}")
        return None

# Process each coin
with PdfPages(os.path.join(results_dir, 'prediction_report.pdf')) as pdf:
    # Create progress bar
    pbar = tqdm(coin_mapping.items(), total=len(coin_mapping), desc="Processing coins")
    
    for coin, coin_id in pbar:
        try:
            # Update progress bar description
            pbar.set_description(f"Processing {coin}")
            
            # Check memory usage before processing each coin
            check_memory_threshold()
            
            print(f"\nProcessing {coin}...")
            print(f"Current memory usage: {get_memory_usage():.2f} MB")
            
            # Read historical data
            historical_file = f'Historical_Data/coin_{coin}.csv'
            if not os.path.exists(historical_file):
                print(f"Historical file not found for {coin}, skipping...")
                continue
                
            historical_df = pd.read_csv(historical_file, parse_dates=['Date'])
            historical_df = historical_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Validate historical data
            if historical_df.isnull().any().any():
                print(f"Missing values in historical data for {coin}, skipping...")
                continue
                
            print(f"Historical data shape: {historical_df.shape}")
            
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
            
            # Split data first
            split_date = df['Date'].max() - timedelta(days=180)
            train = df[df['Date'] < split_date]
            test = df[df['Date'] >= split_date]
            
            if len(train) < 100 or len(test) < 30:  # Minimum data requirements
                print(f"Insufficient data for {coin}, skipping...")
                continue
            
            # Create features separately for train and test
            train = create_features(train, is_training=True)
            test = create_features(test, is_training=False)
            
            # Get price range and corresponding configuration
            price_range = get_price_range(coin_id)
            config = range_config[price_range]
            
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
            
            if not selected_features:
                print(f"No features selected for {coin}, using all features...")
                selected_features = config['features']
            
            # Use only selected features
            X_train_selected = X_train[:, [config['features'].index(f) for f in selected_features]]
            X_test = feature_scaler.transform(test[config['features']])
            X_test_selected = X_test[:, [config['features'].index(f) for f in selected_features]]
            y_test = test['Close'].values
            
            # Train model with selected features and range-specific parameters
            model = create_ensemble_model(price_range)
            model.fit(X_train_selected, y_train)
            
            # Store feature importance (using RandomForest's importance as reference)
            rf_model = RandomForestRegressor(**config['model_params'])
            rf_model.fit(X_train_selected, y_train)
            importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_importance[coin_id] = importance
            
            # Make predictions
            preds_scaled = model.predict(X_test_selected)
            preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            
            # Store results
            results[coin_id] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'Selected_Features': selected_features,
                'Price_Range': price_range
            }
            
            # Create plots
            create_plots(coin_id, test, y_test, preds, price_range, feature_importance, pdf)
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'MAPE': f"{mape:.2f}%",
                'RMSE': f"{rmse:.2f}",
                'Memory': f"{get_memory_usage():.0f}MB"
            })
            
            # Clear memory after processing each coin
            clear_memory()
            
        except Exception as e:
            print(f"Error processing {coin}: {str(e)}")
            clear_memory()  # Clear memory even if there's an error
            continue
        
        time.sleep(2)  # Respect API rate limits
    
    # Create summary plots at the end
    results_df = pd.DataFrame(results).T
    create_summary_plots(results_df, pdf)

# Save results to CSV
results_df.to_csv(os.path.join(results_dir, 'prediction_metrics.csv'))

# Print summary by price range
print("\nPrediction Metrics Summary by Price Range:")
for range_name in ['high', 'mid', 'low']:
    range_results = results_df[results_df['Price_Range'] == range_name]
    print(f"\n{range_name.upper()} Price Range:")
    print(range_results[['MAE', 'MSE', 'RMSE', 'MAPE']].describe())