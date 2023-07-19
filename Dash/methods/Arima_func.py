from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def Arima(historical_data, pred_days):
    """
    Führt mittel ARIMA-Methode eine Kursprognose aus
    :param historical_data: historische Daten 
    :param pred_days: Anz. zu prognostizierender Tage
    :return forecast_df (DF der Vorhersage)
    :return fitted_values (angepasste Trainingsdaten)
    """
    
    model = ARIMA(historical_data, order = (0, 1, 0)) #Definiert das Modell
    model_fit = model.fit() #Trainiert das Modell
    forecasts = model_fit.forecast(steps = pred_days) #Vorhersage
    forecast_df = pd.DataFrame(forecasts.tolist(), columns = ["Close"]) #Erstellt DF der Vorhersage
    fitted_values = model_fit.fittedvalues #Speichert angepasste Trainingsdaten
    return forecast_df, fitted_values
