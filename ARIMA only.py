import yfinance as yf
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

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

best_mae = float('inf')  # Initialize best MAE as infinity
best_params = None  # Initialize best hyperparameters as None

# Hyperparameter tuning loop
for p in range(3):  # Try different values for p (order of autoregressive part)
    for d in range(3):  # Try different values for d (order of differencing)
        for q in range(3):  # Try different values for q (order of moving average part)
            # Fit an ARIMA model on the training data with the current hyperparameters
            model = ARIMA(train_data, order=(p, d, q))
            model_fit = model.fit()

            # Predict the closing prices for the desired prediction range
            y_pred = model_fit.predict(start=prediction_start_date, end=prediction_end_date)

            # Calculate the mean absolute error (MAE) of the predictions
            mae = np.mean(np.abs(y_pred - prediction_data.values.flatten()))

            # Check if the current hyperparameters provide a better MAE
            if mae < best_mae:
                best_mae = mae
                best_params = (p, d, q)

# Fit the ARIMA model with the best hyperparameters on the entire training data
best_model = ARIMA(df, order=best_params)
best_model_fit = best_model.fit()

# Predict the closing prices for the desired prediction range using the best model
best_y_pred = best_model_fit.predict(start=prediction_start_date, end=prediction_end_date)

# Print the best hyperparameters and the corresponding MAE
print(f"Best Hyperparameters: (p={best_params[0]}, d={best_params[1]}, q={best_params[2]})")
print(f"Best Mean Absolute Error (MAE): {best_mae}")
print("Predicted closing prices for the desired prediction range using the best model:")
print(best_y_pred)
