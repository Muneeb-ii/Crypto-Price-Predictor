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