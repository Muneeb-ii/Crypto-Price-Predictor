# Cryptocurrency Price Predictor

A machine learning model that predicts cryptocurrency prices using historical data and technical indicators. The model combines historical price data with recent data from CoinGecko API to make predictions.

## Features

- Predicts prices for 23 different cryptocurrencies
- Uses both historical data and real-time data from CoinGecko
- Implements various technical indicators:
  - Moving Averages (MA5, MA10)
  - Price Changes and Momentum
  - Volatility Indicators
  - High-Low Range Analysis
- Generates performance metrics (MAE, MSE, RMSE, MAPE)
- Creates visualization plots for predictions

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
   - Train the model
   - Generate predictions
   - Save results and plots

## Output

The script generates:
- `prediction_metrics.csv`: Contains MAE, MSE, RMSE, and MAPE for each cryptocurrency
- `predictions_[coin].png`: Plot of actual vs predicted prices for each coin
- `rmse_comparison.png`: Comparison of RMSE across all cryptocurrencies

## Data Structure

Historical data CSV files should contain the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Model Details

- Uses Random Forest Regressor
- Features include:
  - Previous day's close price
  - Price changes
  - Moving averages
  - Volatility measures
  - Momentum indicators
- Train/Test split: Last 6 months used for testing

## Notes

- The CoinGecko API has a rate limit of 30 calls per minute
- The script includes a 2-second delay between API calls to respect rate limits
- Historical data should be in CSV format with appropriate column names

## Future Improvements

- Add more technical indicators
- Implement cross-validation
- Add feature importance analysis
- Include prediction intervals
- Add support for more cryptocurrencies 