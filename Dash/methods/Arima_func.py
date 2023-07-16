from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def Arima(historical_data, pred_days):
    model = ARIMA(historical_data, order = (0, 1, 0))
    model_fit = model.fit()
    forecasts = model_fit.forecast(steps=pred_days)
    forecast_df = pd.DataFrame(forecasts.tolist(), columns=["Close"])
    fitted_values = model_fit.fittedvalues
    return forecast_df, fitted_values