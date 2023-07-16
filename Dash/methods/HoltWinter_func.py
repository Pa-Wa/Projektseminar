from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

def HoltWinter(historical_data, pred_days): 
    model = ExponentialSmoothing(historical_data, trend = "add",
                                  seasonal = "add", initialization_method = "legacy-heuristic", seasonal_periods = 12)
    model_fit = model.fit()
    fitted_values = model_fit.fittedvalues
    forecast = model_fit.forecast(steps=pred_days)
    forecast_df = pd.DataFrame(forecast, columns=["Close"])
    return forecast_df, fitted_values
