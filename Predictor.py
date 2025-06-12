import pandas as pd
import glob, os

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