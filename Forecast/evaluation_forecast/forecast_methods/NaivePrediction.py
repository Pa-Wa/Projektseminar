import pandas as pd

def NaivePrediction(historical_data):
    """
    Führt eine naive Prognose durch. D.h. der letzte bekannte Kurs wird für die vorherzusagenden Tage konstant gehalten
    :param historical_data: Historische Daten
    :return forecast_df (DF der Vorhersage)
    """
    forecast = []
    prices = historical_data["Close"].tolist()
    last_price = prices[-1]
    for i in range(10):
        forecast.append(last_price)
    forecast_df = pd.DataFrame(forecast, columns=["Predicted Price"])
    return forecast_df