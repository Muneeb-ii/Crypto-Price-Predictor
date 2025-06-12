import pandas as pd
import glob, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    
    # Create features
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA5'] = df['Close'].rolling(5).mean().shift(1)
    
    # Drop early rows with NaNs
    df.dropna(inplace=True)
    
    # Define features and target
    features = ['Prev_Close', 'Open', 'High', 'Low', 'Volume', 'MA5']
    X = df[features]
    y = df['Close']
    
    # Split data
    split_date = pd.Timestamp('2020-01-01')
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    
    if len(train) == 0 or len(test) == 0:
        print(f"Skipping {coin} - insufficient data")
        continue
    
    X_train, y_train = train[features], train['Close']
    X_test, y_test = test[features], test['Close']
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    preds = model.predict(X_test)
    
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