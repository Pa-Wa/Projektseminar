import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def arima_regression(stock_symbol):
    # Download stock data
    stock_data = yf.download(stock_symbol, start=pd.Timestamp.now().date() - pd.DateOffset(years=2),
                             end='2023-05-14', progress=False)

    # Only business days
    stock_data = stock_data.asfreq(BDay())

    # Split training data (2 years) and testing data (01-14.05.2023)
    # Only business days so not 14 days but only 10
    training_data = stock_data[:-10]
    testing_data = stock_data[-10:]

    # Fit ARIMA model
    model = ARIMA(training_data['Close'], order=(1, 0, 0))
    model_fit = model.fit()

    # Generate predictions
    predictions = model_fit.predict(start=testing_data.index[0], end=testing_data.index[-1])

    # Calculate mean squared error
    mse = mean_squared_error(testing_data['Close'], predictions)

    # Plot actual values and predictions for the specified range
    plt.plot(testing_data.index, testing_data['Close'], label='Actual')
    plt.plot(testing_data.index, predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('ARIMA Regression - {} Stock (May 1st to May 14th, 2023)'.format(stock_symbol))
    plt.legend()
    plt.text(testing_data.index[-1], testing_data['Close'].iloc[-1], 'MSE: {:.2f}'.format(mse),
             verticalalignment='bottom', horizontalalignment='right')
    plt.show()

    print('Mean Squared Error:', mse)

# Main program
stock_symbol = input('Enter stock abbreviation (e.g., AAPL, MSFT): ')
arima_regression(stock_symbol)
