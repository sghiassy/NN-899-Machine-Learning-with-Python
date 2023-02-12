import pandas as pd
import quandl
import math

print("Fetching Google Stock Prices")

df = quandl.get("WIKI/GOOGL", authtoken="i69xthGji4B3cjxRQ1dJ")

print(df.keys())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

print(df)

# High Low Percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

# Percent Change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

print(df)
print("Cleaning up DF")

# Clean up the data frame
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df)

#### Start of 2nd Video: https://www.youtube.com/watch?v=lN5jesocJjk&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=3

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) # days to shift out the data column

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())