# Cryptocurrency Price Predictor

A comprehensive cryptocurrency price prediction system that implements both Machine Learning (ML) and Deep Learning (DL) approaches. The system uses historical data from Kaggle and recent data from the CoinGecko API to make predictions for 23 different cryptocurrencies.

## Data Sources

- **Historical Data**: Kaggle dataset containing historical price data for 23 cryptocurrencies up until 2021
- **Recent Data**: CoinGecko API for up-to-date price information (limited to last 365 days)
- **Data Gap**: There is a gap between the historical data (ending 2021) and recent data (last 365 days)
- **Data Structure**: Each cryptocurrency's data includes Date, Open, High, Low, Close, and Volume

## Models

### 1. Machine Learning Model (ML)

#### Architecture
- **Ensemble Model**: Voting Regressor combining:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
- **Feature Selection**:
  - Recursive Feature Elimination (RFE) with Random Forest
  - LASSO with cross-validation
  - Automatic selection of most important features for each coin

#### Features and Parameters
- **Price Range-Specific Configuration**:
  - High-value coins (Bitcoin, Ethereum, WrappedBitcoin):
    - Advanced technical indicators (MA5-MA50, MACD, RSI, Bollinger Bands)
    - Complex volatility measures
    - Deeper trees (max_depth=15) and more estimators (n_estimators=200)
  - Mid-value coins (BinanceCoin, Solana, Cardano, etc.):
    - Standard technical indicators
    - Basic moving averages
    - Balanced complexity (max_depth=10, n_estimators=150)
  - Low-value coins:
    - Essential price metrics
    - Basic moving averages
    - Simpler models (max_depth=8, n_estimators=100)

#### Training Process
- **Data Split**: Last 6 months used for testing
- **Memory Management**:
  - Automatic memory cleanup between coins
  - Memory threshold monitoring (1000MB)
  - Garbage collection after each coin
- **API Handling**:
  - 2-second delay between CoinGecko API calls
  - Rate limit: 30 calls per minute
  - Automatic retry on failure

#### Performance Metrics (ML Model)
- **Best Performing Coins**:
  - USDCoin: 0.005% MAPE
  - Tether: 0.009% MAPE
  - Aave: 1.25% MAPE
  - Ethereum: 1.02% MAPE

- **Challenging Predictions**:
  - Bitcoin: 1.83% MAPE
  - WrappedBitcoin: 2.31% MAPE
  - Tron: 3.97% MAPE
  - Cardano: 2.56% MAPE

### 2. Deep Learning Model (DL)

#### Architecture
- **LSTM Model**: Sequential model with:
  - Two LSTM layers (100 and 50 units)
  - Dropout layers (0.2) for regularization
  - Dense layers for final prediction
- **Training Process**:
  - Uses only historical data (up to 2021)
  - 70-10-20 split for train-validation-test
  - Sequence length of 60 time steps
  - Mixed precision training for better performance

### Model Differences
- **ML Model**: Uses both historical and recent data, making predictions for the last 6 months
- **DL Model**: Uses only historical data (up to 2021) due to the data gap between historical and recent data
- **Reason for Difference**: The large gap in data (2021 to present) makes it challenging for the LSTM model to learn meaningful patterns across this discontinuity

### Future Improvements
- Working on gathering the missing data between 2021 and present
- Planning to update the DL model once the complete dataset is available
- Aiming to achieve consistent performance across both models with complete data

#### Performance Metrics (LSTM Model)
- **Best Performing Coins**:
  - USDCoin: 0.05% MAPE
  - Tether: 0.15% MAPE
  - Bitcoin: 4.70% MAPE
  - Monero: 4.35% MAPE

- **Challenging Predictions**:
  - ChainLink: 50.26% MAPE
  - BinanceCoin: 20.88% MAPE
  - Cardano: 17.09% MAPE
  - Dogecoin: 18.10% MAPE

## Model Strengths and Weaknesses

### ML Model Strengths
1. Price range-specific optimization
2. Feature selection using RFE and LASSO
3. Better handling of different price scales
4. More interpretable results
5. Faster training time

### ML Model Weaknesses
1. Less effective with long-term dependencies
2. May miss complex patterns in price movements
3. Requires manual feature engineering

### LSTM Model Strengths
1. Better at capturing long-term dependencies
2. Can learn complex patterns automatically
3. More suitable for time series data
4. No need for manual feature engineering

### LSTM Model Weaknesses
1. Longer training time
2. More sensitive to hyperparameters
3. Requires more data for optimal performance
4. Less interpretable results

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
  - pandas>=1.5.0
  - numpy>=1.21.0
  - scikit-learn>=1.0.0
  - tensorflow>=2.8.0
  - matplotlib>=3.10.3
  - pycoingecko>=3.0.0
  - seaborn>=0.12.0
  - tqdm>=4.65.0

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

1. Ensure historical data is in the `Historical_Data` directory with files named as `coin_[CryptocurrencyName].csv`

2. Run the ML predictor:
```bash
python predictor_ml.py
```

3. Run the LSTM predictor:
```bash
python predictor_dl.py
```

## Output

Both models generate results in their respective directories:
- ML Model: `results/randomforest_prediction_report/`
  - `prediction_metrics.csv`: Performance metrics and selected features
  - `prediction_report.pdf`: Visualizations and analysis
  - Model checkpoints for each coin

- LSTM Model: `results/lstm_prediction_report/`
  - `prediction_metrics.csv`: Performance metrics
  - `prediction_report.pdf`: Visualizations and analysis
  - `{coin}_best_model.keras`: Best model for each coin

## Notes

- **API Limitations**:
  - CoinGecko API rate limit: 30 calls per minute
  - 2-second delay between API calls
  - Automatic handling of rate limits and retries

- **Model Training**:
  - Both models use early stopping to prevent overfitting
  - ML model: Feature selection before training
  - LSTM model: Mixed precision training for better performance

- **Results Storage**:
  - ML Model: `results/randomforest_prediction_report/`
  - LSTM Model: `results/lstm_prediction_report/`
  - Each model maintains separate metrics and visualizations
