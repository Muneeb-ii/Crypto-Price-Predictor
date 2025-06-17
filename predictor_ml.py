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
import gc
import psutil
from tqdm import tqdm
import seaborn as sns

# Constants
RESULTS_DIR = 'results/randomforest_prediction_report'
HISTORICAL_DATA_DIR = 'Historical_Data'
MIN_TRAIN_SIZE = 100
MIN_TEST_SIZE = 30
TEST_PERIOD_DAYS = 180
MEMORY_THRESHOLD_MB = 1000

def check_directories():
    """Check if required directories exist and create them if needed"""
    required_dirs = [RESULTS_DIR, HISTORICAL_DATA_DIR]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def clear_memory():
    """Clear memory by running garbage collection"""
    gc.collect()

def check_memory_threshold(threshold_mb=MEMORY_THRESHOLD_MB):
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
        
        prices = np.array(data['prices'])
        volumes = np.array(data['total_volumes'])
        
        if len(prices) == 0 or len(volumes) == 0:
            print(f"Empty data arrays for {coin_id}")
            return None
        
        df = pd.DataFrame({
            'Date': [datetime.fromtimestamp(ts/1000) for ts in prices[:, 0]],
            'Close': prices[:, 1],
            'Volume': volumes[:, 1]
        })
        
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close']
        df['Low'] = df['Close']
        
        if df['Close'].isnull().any() or df['Volume'].isnull().any():
            print(f"Missing values in data for {coin_id}")
            return None
        
        return df.dropna()
    
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
    """Create an ensemble model based on price range"""
    if price_range == 'high':
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8
            )),
            ('xgb', XGBRegressor(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8
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
    
    # Price-based features
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['VWAP_MA5'] = df['VWAP'].rolling(window=5, min_periods=1).mean().shift(1)
    df['VWAP_MA20'] = df['VWAP'].rolling(window=20, min_periods=1).mean().shift(1)
    df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # Basic price features
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
    
    # Volatility and momentum
    df['Volatility'] = df['Price_Change'].rolling(window=10, min_periods=1).std().shift(1)
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_MA5'] = df['Momentum'].rolling(5, min_periods=1).mean().shift(1)
    df['Momentum_MA20'] = df['Momentum'].rolling(20, min_periods=1).mean().shift(1)
    
    # Technical indicators
    df['RSI'] = calculate_rsi(df['Close']).shift(1)
    df['RSI_MA5'] = df['RSI'].rolling(5, min_periods=1).mean().shift(1)
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd.shift(1)
    df['MACD_Signal'] = signal.shift(1)
    df['MACD_Hist'] = hist.shift(1)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean().shift(1)
    df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std().shift(1)
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Price volatility
    df['Price_Volatility_5d'] = df['Price_Change'].rolling(window=5, min_periods=1).std().shift(1)
    df['Price_Volatility_20d'] = df['Price_Change'].rolling(window=20, min_periods=1).std().shift(1)
    
    return df.dropna()

def create_plots(coin, test, y_test, preds, price_range, feature_importance, pdf):
    """Create and save plots to PDF"""
    fig = plt.figure(figsize=(15, 10))
    
    # Price Predictions
    plt.subplot(2, 2, 1)
    plt.plot(test['Date'], y_test, label='Actual', color='blue')
    plt.plot(test['Date'], preds, label='Predicted', color='red')
    plt.title(f'{coin} Price Prediction (Last 6 Months) - {price_range} range')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Feature Importance
    plt.subplot(2, 2, 2)
    importance = feature_importance[coin]
    plt.barh(importance['Feature'], importance['Importance'])
    plt.title(f'{coin} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
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

def process_coin(coin, coin_id, historical_df, recent_df):
    """Process a single cryptocurrency"""
    try:
        print(f"\nProcessing {coin}...")
        print(f"Current memory usage: {get_memory_usage():.2f} MB")
        
        # Combine and validate data
        if recent_df is None:
            df = historical_df
        else:
            df = pd.concat([historical_df, recent_df])
            df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # Ensure we have enough data
        if len(df) < MIN_TRAIN_SIZE + MIN_TEST_SIZE:
            print(f"Insufficient data for {coin}, skipping...")
            return None
            
        # Split data ensuring no overlap
        split_date = df['Date'].max() - timedelta(days=TEST_PERIOD_DAYS)
        train = df[df['Date'] < split_date].copy()  # Use copy to prevent any data leakage
        test = df[df['Date'] >= split_date].copy()
        
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
        
        # Create features separately to prevent any data leakage
        print(f"Creating features for {coin}...")
        train = create_features(train, is_training=True)
        test = create_features(test, is_training=False)
        
        # Verify no data leakage in features
        train_dates = set(train['Date'])
        test_dates = set(test['Date'])
        if train_dates.intersection(test_dates):
            raise ValueError(f"Data leakage detected in {coin}: overlapping dates between train and test")
        
        # Get price range and configuration
        price_range = get_price_range(coin_id)
        config = range_config[price_range]
        
        # Scale features and target using only training data
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        # Fit scalers on training data only
        X_train_df = train[config['features']]
        y_train_df = train[['Close']]
        
        # Transform training data
        X_train = pd.DataFrame(
            feature_scaler.fit_transform(X_train_df),
            columns=config['features'],
            index=X_train_df.index
        )
        y_train = target_scaler.fit_transform(y_train_df).ravel()
        
        # Transform test data using training scalers
        X_test_df = test[config['features']]
        X_test = pd.DataFrame(
            feature_scaler.transform(X_test_df),  # Only transform, don't fit
            columns=config['features'],
            index=X_test_df.index
        )
        y_test = test['Close'].values  # Keep original scale for metrics
        
        # Feature selection using only training data
        print(f"Performing feature selection for {coin}...")
        rfe = RFE(
            estimator=RandomForestRegressor(**config['model_params']),
            n_features_to_select=min(10, len(config['features'])),
            step=1
        )
        rfe.fit(X_train, y_train)
        
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_train, y_train)
        
        rfe_features = [f for f, s in zip(config['features'], rfe.support_) if s]
        lasso_features = [f for f, c in zip(config['features'], lasso.coef_) if abs(c) > 0]
        selected_features = list(set(rfe_features + lasso_features))
        
        if not selected_features:
            selected_features = config['features']
        
        # Use selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Train model
        print(f"Training model for {coin}...")
        model = create_ensemble_model(price_range)
        model.fit(X_train_selected, y_train)
        
        # Store feature importance
        rf_model = RandomForestRegressor(**config['model_params'])
        rf_model.fit(X_train_selected, y_train)
        importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Make predictions
        print(f"Making predictions for {coin}...")
        preds_scaled = model.predict(X_test_selected)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
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
            'Selected_Features': selected_features,
            'Price_Range': price_range,
            'test_data': test,
            'y_test': y_test,
            'preds': preds,
            'feature_importance': importance
        }
        
    except Exception as e:
        print(f"Error processing {coin}: {str(e)}")
        return None

if __name__ == "__main__":
    # Initialize
    check_directories()
    results = {}
    feature_importance = {}
    
    # Process each coin
    with PdfPages(os.path.join(RESULTS_DIR, 'prediction_report.pdf')) as pdf:
        pbar = tqdm(coin_mapping.items(), total=len(coin_mapping), desc="Processing coins")
        
        for coin, coin_id in pbar:
            try:
                # Check memory
                check_memory_threshold()
                
                # Load historical data
                historical_file = os.path.join(HISTORICAL_DATA_DIR, f'coin_{coin}.csv')
                if not os.path.exists(historical_file):
                    print(f"Historical file not found for {coin}, skipping...")
                    continue
                
                historical_df = pd.read_csv(historical_file, parse_dates=['Date'])
                historical_df = historical_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                if historical_df.isnull().any().any():
                    print(f"Missing values in historical data for {coin}, skipping...")
                    continue
                
                # Fetch recent data
                recent_df = fetch_coingecko_data(coin_id)
                
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
                    'Selected_Features': result['Selected_Features'],
                    'Price_Range': result['Price_Range']
                }
                feature_importance[coin_id] = result['feature_importance']
                
                # Create plots
                create_plots(coin_id, result['test_data'], result['y_test'], 
                           result['preds'], result['Price_Range'], 
                           feature_importance, pdf)
                
                # Update progress bar
                pbar.set_postfix({
                    'MAPE': f"{result['MAPE']:.2f}%",
                    'RMSE': f"{result['RMSE']:.2f}",
                    'Memory': f"{get_memory_usage():.0f}MB"
                })
                
            except Exception as e:
                print(f"Error processing {coin}: {str(e)}")
                continue
            finally:
                clear_memory()
                time.sleep(2)  # Respect API rate limits
        
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