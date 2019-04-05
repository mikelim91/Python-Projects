import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import seaborn as sns
import math

# # Introduction


df = pd.read_csv(r'C:\Users\mikel\OneDrive\Desktop\Python\Datasets\sphist.csv')
print(len(df))

# Convert Date col to Pandas Date Format
df['Date'] = pd.to_datetime(df['Date'])

# Boolean Condition (from datetime import datetime)
a = df["Date"] > datetime(year=2015, month=4, day=1)

df = df.sort_values(by='Date', ascending=True)

print('mean is: ' + str(df['Close'][:4].mean()))


df['day_5'] = 0
df['day_30'] = 0
df['day_365'] = 0
print(df[:2])


# Calculates rolling mean and shifts index position by 1
df['day_5'] = df['Close'].rolling(5).mean()
df['day_5'] = df['day_5'].shift()
df['day_30'] = df['Close'].rolling(30).mean()
df['day_30'] = df['day_30'].shift()
df['day_365'] = df['Close'].rolling(365).mean()
df['day_365'] = df['day_365'].shift()
print(df.head())


# Solution for w/ iterrows (takes longer)
# for row, index in df.iterrows():
#         day_5 = df["day_5"][index]
#         day_30 = df['day_30'][index]
#         day_365 = df["day_365"][index]
#         print(row, index)


# Drop NaN values and no dates before 5/2/1951
df = df[(df['Date'] > datetime(year=1951, month=1, day=2))]
df = df.dropna(axis=0)
print(len(df))
print(df.head())

# Training Data
a = df['Date'] < datetime(year=2013, month=1, day=1)
b = df['Date'] >= datetime(year=2013, month=1, day=1)
train = df[a]
test = df[b]
print(train.head(10))
print(test.head(10))

# Initialize an instance of the LinearRegression class.
# Train a linear regression model, using the train Dataframe.
# Leave out all of the original columns (Close, High, Low, Open, Volume, Adj Close, Date) when training model.
# These all contain knowledge of the future that should not be fed into the model. Use the Close column as the target.
# Make predictions for the Close column of the test data, using the same columns for training as you did with train.
# Compute the error between the predictions and the Close column of
# Linear Regression

lr = LinearRegression()
lr.fit(train[['day_5']], train[['Close']])
prediction = lr.predict(test[['day_5']])
mse = mean_squared_error(test['Close'], prediction)
print(mse)

lr = LinearRegression()
lr.fit(train[['day_30']], train[['Close']])
prediction = lr.predict(test[['day_30']])
mse = mean_squared_error(test['Close'], prediction)
print(mse)

lr = LinearRegression()
lr.fit(train[['day_365']], train[['Close']])
prediction = lr.predict(test[['day_365']])
mse = mean_squared_error(test['Close'], prediction)
print(mse)


# Additional indicators
lr = LinearRegression()
lr.fit(train[['day_5', 'day_30']], train[['Close']])
prediction = lr.predict(test[['day_5', 'day_30']])
mse = mean_squared_error(test['Close'], prediction)
print(mse)

lr = LinearRegression()
lr.fit(train[['day_5', 'day_30', 'day_365']], train[['Close']])
prediction = lr.predict(test[['day_5', 'day_30', 'day_365']])
mse = mean_squared_error(test['Close'], prediction)
print(mse)

