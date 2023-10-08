import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
stock_data = pd.read_csv('C:\\Users\\titanic\\apple\\HistoricalQuotes.csv')
stock_data.head()
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2021, 10, 1)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
filtered_apple_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
print(filtered_apple_data.head())
print(stock_data.columns)
print(stock_data.head())
plt.figure(figsize=(15, 8))
plt.title('Stock Prices History')
plt.plot(stock_data[' Close/Last'])  # Use the correct column name
plt.xlabel('Date')
plt.ylabel('Prices ($)')
plt.show()
close_prices = stock_data[' Close/Last']  # Use the correct column name
values = close_prices.values
training_data_len = math.ceil(len(values) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))
x_train = []
y_train = []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 1, epochs=3)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train, label='Train')
plt.plot(validation[' Close/Last'], label='Val')  # Use the correct column name
plt.plot(validation['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.show()
import statsmodels.api as sm
num_samples, time_steps, num_features = x_train.shape
for i in range(num_samples):
    sequence = x_train[i, :, 0]  # Extract the time series sequence
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(sequence, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(sequence, lags=10, ax=ax2)  # Use a smaller number of lags
    plt.show()

import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'stock_data' with Apple data, and you want to plot the volume data.
# Make sure you have loaded your data correctly.
import pandas as pd

# Load your data from a CSV file (replace 'your_data.csv' with your actual data file)
stock_data = pd.read_csv('C:\\Users\\titanic\\apple\\HistoricalQuotes.csv')

# Now, you can use the 'stock_data' DataFrame in your plotting code.

# Extract the necessary data from your DataFrame (modify as needed)
dates = stock_data['Date']
volume = stock_data[' Volume']

# Create a bar graph
plt.figure(figsize=(12, 6))  # Set the figure size (optional)

# Plot the bars
plt.bar(dates, volume, color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Apple Stock Volume')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the graph
plt.show()

import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'stock_data' with Apple data, and you want to plot the closing prices.
# Make sure you have loaded your data correctly.
stock_data = pd.read_csv('C:\\Users\\titanic\\apple\\HistoricalQuotes.csv')
# Extract the necessary data from your DataFrame (modify as needed)
dates = stock_data['Date']
closing_prices = stock_data[' Close/Last']

# Create a line plot
plt.figure(figsize=(12, 6))  # Set the figure size (optional)

# Plot the line
plt.plot(dates, closing_prices, label='Closing Prices', color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.title('Apple Stock Closing Prices')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Add a legend (optional)
plt.legend()

# Show the graph
plt.show()

import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'stock_data' with Apple data, and you want to plot a scatter plot of opening vs. closing prices.
# Make sure you have loaded your data correctly.
stock_data = pd.read_csv('C:\\Users\\titanic\\apple\\HistoricalQuotes.csv')
# Extract the necessary data from your DataFrame (modify as needed)
opening_prices = stock_data[' Open']
closing_prices = stock_data[' Close/Last']

# Create a scatter plot
plt.figure(figsize=(10, 6))  # Set the figure size (optional)

# Plot the scatter points
plt.scatter(opening_prices, closing_prices, color='green', alpha=0.5)  # alpha sets transparency

# Add labels and title
plt.xlabel('Opening Prices')
plt.ylabel('Closing Prices')
plt.title('Scatter Plot of Opening vs. Closing Prices')

# Show the graph
plt.show()
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'stock_data' with Apple data, and you want to create a histogram of closing prices.
# Make sure you have loaded your data correctly.

# Extract the necessary data from your DataFrame (modify as needed)
closing_prices = stock_data[' Close/Last']

# Create a histogram
plt.figure(figsize=(10, 6))  # Set the figure size (optional)

# Plot the histogram
plt.hist(closing_prices, bins=30, color='purple', edgecolor='black')

# Add labels and title
plt.xlabel('Closing Prices')
plt.ylabel('Frequency')
plt.title('Histogram of Apple Stock Closing Prices')

# Show the graph
plt.show()


