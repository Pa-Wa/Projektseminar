import yfinance as yf
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

# Get user input for stock symbol
symbol = input("Enter stock symbol: ")

# Download historical data for the given stock
stock_data = yf.download(symbol, period="max")

# Create a new dataframe with only the closing prices
df = stock_data[['Close']].reset_index()

# Set the index of the dataframe to the date column
df.set_index('Date', inplace=True)

# Define the desired date ranges
train_start_date = '2023-01-01'
train_end_date = '2023-05-31'
prediction_start_date = '2023-05-01'
prediction_end_date = '2023-05-31'

# Filter the data based on the date ranges
train_data = df.loc[train_start_date:train_end_date]
prediction_data = df.loc[prediction_start_date:prediction_end_date]

# Fit an ARIMA model on the training data
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

# Predict the closing prices for the desired prediction range
y_pred = model_fit.predict(start=prediction_start_date, end=prediction_end_date)

# Calculate the mean absolute error (MAE) of the predictions
mae = np.mean(np.abs(y_pred - prediction_data.values.flatten()))

# Print the MAE
print(f"Mean Absolute Error (MAE): {mae}")

# Create a figure and axis for the graph
fig, ax = plt.subplots()

# Plot the actual and predicted closing prices
ax.plot(prediction_data.index, prediction_data.values, label='Actual')
ax.plot(prediction_data.index, y_pred, label='Predicted')

# Set labels, title, legend and display graph
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.set_title('Actual vs. Predicted Closing Prices')
plt.xticks(rotation=45)
ax.legend()
plt.show()
