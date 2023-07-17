from pmdarima.arima import auto_arima
import pandas as pd

"""
Datei wurde auch zur Analyse der Parameter genutzt.
Dazu einfach in Main-Datei den Zeithorizont ändern.
"""

def ARIMA(historical_data):
    """
    Führt mittel ARIMA-Methode eine Kursprognose aus
    :param historical_data: historische Daten 
    :return forecast_df (DF der Vorhersage)
    """
    
    model = auto_arima(historical_data, seasonal = True, stepwise = True, m = 6, method = "bfgs", maxiter = 20, suppress_warnings = True) #Definiert das Modell
    forecasts = model.predict(n_periods = 10) #Vorhersage
    forecast_df = pd.DataFrame(forecasts, columns=["Predicted Price"]) #Erstellt DF der Vorhersage
    return forecast_df
