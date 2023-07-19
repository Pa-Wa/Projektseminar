from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

def HoltWinters(historical_data, pred_days): 
    """
    Führt eine Kursprognose anhand der Holt-Winters Methode aus
    :param historical_data: Historische Daten
    :param pred_days: Anz. vorherzusagender Tage
    :return: forecast_df (Prognosen im DF abgespeichert)
    :return: fitted_values (Angepasste Trainings-Daten)
    """
    
    model = ExponentialSmoothing(historical_data, trend = "add",
                                  seasonal = "add", initialization_method = "legacy-heuristic", seasonal_periods = 12) #Definiert Modell
    model_fit = model.fit() #Trainiert Modell
    fitted_values = model_fit.fittedvalues #Speichert angepasste, trainierte Daten ab
    forecast = model_fit.forecast(steps = pred_days) #Sagt Kursdaten vorher
    forecast_df = pd.DataFrame(forecast, columns = ["Close"]) #Speichert Kursdaten in einem DF ab
    return forecast_df, fitted_values
