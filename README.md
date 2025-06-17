# Cryptocurrency Price Predictor

A comprehensive cryptocurrency price prediction system that implements both Machine Learning (ML) and Deep Learning (DL) approaches. The system uses historical data from Kaggle and recent data from the CoinGecko API to make predictions for 23 different cryptocurrencies.

## Data Sources

- **Historical Data**: Kaggle dataset containing historical price data for 23 cryptocurrencies
- **Recent Data**: CoinGecko API for up-to-date price information
- **Data Structure**: Each cryptocurrency's data includes Date, Open, High, Low, Close, and Volume

## Models

### 1. Machine Learning Model (ML)

#### Architecture
- **Ensemble Model**: Voting Regressor combining:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor

#### Features
- **Price Range-Specific Features**:
  - High-value coins (Bitcoin, Ethereum, WrappedBitcoin):
    - Advanced technical indicators
    - Extended moving averages (MA5-MA50)
    - Complex volatility measures
  - Mid-value coins (BinanceCoin, Solana, Cardano, etc.):
    - Standard technical indicators
    - Basic moving averages
    - Volume analysis
  - Low-value coins:
    - Essential price metrics
    - Basic moving averages
    - Simplified indicators

#### Performance Metrics (ML Model)
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

### 2. Deep Learning Model (LSTM)

#### Architecture
- **Sequential LSTM Model**:
  - Input Layer: 60 time steps
  - LSTM Layer 1: 100 units with return sequences
  - Dropout Layer: 0.2
  - LSTM Layer 2: 50 units
  - Dropout Layer: 0.2
  - Dense Layer: 25 units
  - Output Layer: 1 unit

#### Training Parameters
- Batch Size: 32
- Epochs: 100 (with early stopping)
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Mean Squared Error

#### Performance Metrics (LSTM Model)
- **Overall Statistics**:
  - Mean MAPE: 10.60%
  - Median MAPE: 6.53%
  - Best MAPE: 0.05% (USDCoin)
  - Worst MAPE: 50.26% (ChainLink)

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

Both models generate:
- `prediction_metrics.csv`: Performance metrics for each cryptocurrency
- `prediction_report.pdf`: Detailed visualizations and analysis
- Model checkpoints in their respective results directories

## Future Improvements

1. **Model Enhancements**:
   - Implement attention mechanisms in LSTM
   - Add transformer-based models
   - Improve ensemble methods

2. **Feature Engineering**:
   - Add sentiment analysis
   - Include on-chain metrics
   - Integrate macroeconomic indicators

3. **Technical Improvements**:
   - Implement cross-validation
   - Add prediction intervals
   - Improve error handling
   - Optimize memory usage

4. **User Experience**:
   - Add real-time prediction API
   - Create interactive visualizations
   - Implement automated model selection

## Notes

- CoinGecko API has a rate limit of 30 calls per minute
- Scripts include appropriate delays to respect API limits
- Both models use early stopping to prevent overfitting
- Results are saved in separate directories for each model

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

## Data Structure

Historical data CSV files should contain the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Future Improvements

- Fine-tune model parameters for each price range
- Implement time-series cross-validation for more robust evaluation
- Add support for model ensembling (e.g., XGBoost, LightGBM)
- Integrate additional data sources (on-chain, sentiment, macroeconomic)
- Improve handling of outlier coins with high error
- Add prediction intervals and uncertainty estimation
- Support for more cryptocurrencies and new technical indicators 