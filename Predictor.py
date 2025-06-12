import pandas as pd
import glob, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# point to your folder
csv_files = glob.glob('Historical_Data/coin_*.csv')

# Dictionary to store results for each coin
results = {}

for fp in csv_files:
    # Read and prepare data for each coin
    df = pd.read_csv(fp, parse_dates=['Date'])
    coin = os.path.basename(fp).split('_', 1)[1].rsplit('.',1)[0]
    print(f"\nProcessing {coin}...")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create basic features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['MA5'] = df['Close'].rolling(5).mean().shift(1)
    df['MA10'] = df['Close'].rolling(10).mean().shift(1)
    
    # Add volatility and momentum features
    df['Volatility'] = df['Price_Change'].rolling(window=10).std()
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Range_5d'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / df['Close']
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_MA5'] = df['Momentum'].rolling(5).mean()
    
    # Drop early rows with NaNs
    df.dropna(inplace=True)
    
    # Define features and target
    features = [
        'Prev_Close', 'Open', 'High', 'Low', 'Volume', 
        'MA5', 'MA10', 'Price_Change', 'Price_Change_5d',
        'Volatility', 'High_Low_Range', 'Price_Range_5d',
        'Momentum', 'Momentum_MA5'
    ]
    
    # Split data
    split_date = pd.Timestamp('2021-05-01')
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    
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
    plt.title(f'{coin} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predictions_{coin}.png')
    plt.close()

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