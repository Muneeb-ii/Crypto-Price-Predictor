# Cryptocurrency Price Predictor

A machine learning model that predicts cryptocurrency prices using historical data and technical indicators. The model combines historical price data with recent data from CoinGecko API to make predictions.

## Features

- Predicts prices for 23 different cryptocurrencies
- Uses both historical data and real-time data from CoinGecko
- Implements various technical indicators:
  - Moving Averages (MA5, MA10, MA20, MA50)
  - Price Changes and Momentum
  - Volatility Indicators
  - High-Low Range Analysis
  - Bollinger Bands
  - MACD, RSI
- **Price range-specific models:**
  - High-value, mid-value, and low-value coins use different feature sets and model parameters for optimal accuracy
- **Automated feature selection:**
  - Uses RFE (Recursive Feature Elimination) and LASSO to select the most important features for each coin
- Generates performance metrics (MAE, MSE, RMSE, MAPE)
- Creates visualization plots for predictions and RMSE by price range

## Supported Cryptocurrencies

- Aave
- BinanceCoin
- Bitcoin
- Cardano
- ChainLink
- Cosmos
- CryptocomCoin
- Dogecoin
- EOS
- Ethereum
- Iota
- Litecoin
- Monero
- NEM
- Polkadot
- Solana
- Stellar
- Tether
- Tron
- USDCoin
- Uniswap
- WrappedBitcoin
- XRP

## Requirements

- Python 3.7+
- Required packages (see requirements.txt):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - pycoingecko
  - requests

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Crypto-Price-Predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your historical data is in the `Historical_Data` directory with files named as `coin_[CryptocurrencyName].csv`

2. Run the predictor:
```bash
python Predictor.py
```

3. The script will:
   - Load historical data
   - Fetch recent data from CoinGecko
   - Engineer features (including price range-specific features)
   - Drop rows with missing values
   - Select the most important features for each coin using RFE and LASSO
   - Train a price range-specific Random Forest model
   - Generate predictions
   - Save results and plots

## Output

The script generates:
- `prediction_metrics.csv`: Contains MAE, MSE, RMSE, MAPE, selected features, and price range for each cryptocurrency
- `predictions_[coin].png`: Plot of actual vs predicted prices for each coin
- `rmse_comparison_by_range.png`: Comparison of RMSE across all cryptocurrencies, grouped by price range

## Data Structure

Historical data CSV files should contain the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Model Details

- Uses Random Forest Regressor with price range-specific parameters
- Features include:
  - Previous day's close price
  - Price changes (1d, 5d, 20d)
  - Moving averages (MA5, MA10, MA20, MA50)
  - Volatility measures
  - Momentum indicators
  - Technical indicators (MACD, RSI, Bollinger Bands)
  - Volume-based features
- **Price range-specific modeling:**
  - High-value coins use more sophisticated features and deeper models
  - Mid-value and low-value coins use tailored feature sets and model complexity
- **Feature selection:**
  - RFE and LASSO are used to select the most relevant features for each coin before training
- Train/Test split: Last 6 months used for testing

## Notes

- The CoinGecko API has a rate limit of 30 calls per minute
- The script includes a 2-second delay between API calls to respect rate limits
- Historical data should be in CSV format with appropriate column names

## Future Improvements

- Fine-tune model parameters for each price range
- Implement time-series cross-validation for more robust evaluation
- Add support for model ensembling (e.g., XGBoost, LightGBM)
- Integrate additional data sources (on-chain, sentiment, macroeconomic)
- Improve handling of outlier coins with high error
- Add prediction intervals and uncertainty estimation
- Support for more cryptocurrencies and new technical indicators 