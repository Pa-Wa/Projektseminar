from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

def HoltWinters(historical_data): 
    """
    Führt eine Kursprognose anhand der Holt-Winters Methode aus
    :param historical_data: Historische Daten
    :return: forecast_df (Prognosen im DF abgespeichert)
    """
    
    model = ExponentialSmoothing(historical_data, trend = "add", seasonal = "add", initialization_method = "legacy-heuristic",
                                seasonal_periods = 12) #Definiert Modell
    model_fit = model.fit() #bspw. smoothing_trend = 0.2 #Trainiert Modell
    forecast = model_fit.forecast(steps = 10) #Sagt Kursdaten vorher
    forecast_df = pd.DataFrame(forecast, columns = ["Predicted Price"]) #Speichert Kursdaten in einem DF ab
    return forecast_df