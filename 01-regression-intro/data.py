import pandas as pd
import quandl

print("Fetching Google Stock Prices")

df = quandl.get("WIKI/GOOGL", authtoken="i69xthGji4B3cjxRQ1dJ")

print(df.keys())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

print(df)

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

print(df)
print("Cleaning up DF")

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df)