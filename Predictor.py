import pandas as pd
import glob, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# point to your folder
csv_files = glob.glob('Historical_Data/coin_*.csv')

dfs = []
for fp in csv_files:
    df = pd.read_csv(fp, parse_dates=['Date'])
    # extract coin name from filename, e.g. "coin_Bitcoin.csv" → "Bitcoin"
    coin = os.path.basename(fp).split('_', 1)[1].rsplit('.',1)[0]
    df['Coin'] = coin
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(data.head())

# pick a single coin
coin_df = (data
    .query("Coin == 'Bitcoin'")
    .sort_values('Date')
    .reset_index(drop=True)
)

# create lagged & rolling features
coin_df['Prev_Close'] = coin_df['Close'].shift(1)
coin_df['MA5']       = coin_df['Close'].rolling(5).mean().shift(1)

# drop early rows with NaNs
coin_df.dropna(inplace=True)

# define feature matrix X and target y
features = ['Prev_Close','Open','High','Low','Volume','MA5']
X = coin_df[features]
y = coin_df['Close']

# e.g. train on data before 2020-01-01
split_date = pd.Timestamp('2020-01-01')
train = coin_df[coin_df['Date'] < split_date]
test  = coin_df[coin_df['Date'] >= split_date]

X_train, y_train = train[features], train['Close']
X_test,  y_test  = test[features],  test['Close']

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Test MAE: {mae:.2f} USD")

# plot actual vs. predicted
plt.figure(figsize=(10,5))
plt.plot(test['Date'], y_test, label='Actual')
plt.plot(test['Date'], preds, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()