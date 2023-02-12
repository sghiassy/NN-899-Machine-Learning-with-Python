import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

path_prefix = './01-regression-intro/tmp_data/'

print("Fetching Google Stock Prices")
# data = quandl.get('WIKI/GOOGL', authtoken="i69xthGji4B3cjxRQ1dJ")
# data.to_csv(path_prefix + '00.googl.csv')
df = pd.read_csv(path_prefix + '00.googl.csv', index_col='Date', parse_dates=True)

print(df.keys())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df.to_csv(path_prefix + '02.filtered_dataframe.csv', index=True)

print(df)

# High Low Percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

# Percent Change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

print(df)
df.to_csv(path_prefix + '03.added_columns.csv', index=True)
print("Cleaning up DF")

# Clean up the data frame
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
df.to_csv(path_prefix + '04.cleaned_up_columns.csv', index=True)

print(df)

#### Start of 2nd Video: https://www.youtube.com/watch?v=lN5jesocJjk&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=3

print("Start of 2nd Video")

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) # days to shift out the data column

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

print(df.head())

# Start of 3rd Video: https://www.youtube.com/watch?v=r4mwkS2T9aI&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=4

print("Start of 3rd Video")

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_orig = X
X = X[:-forecast_out]
X_lately = X[forecast_out:]
print('X_lately', X_lately)

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_orig, y, test_size=0.2)

print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)
print("\ny_train:")
print(y_train)
print("\ny_test:")
print(y_test)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy of the LinearRegression:", accuracy*100, "%")

# Start of 5th Video: https://www.youtube.com/watch?v=QLVMqwpOLPk&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=5

print("Start of 5th Video")

forecast_set = clf.predict(X_lately)

print("forecast_set", forecast_set)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()