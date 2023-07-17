import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from forecast_methods.HoltWinters import HoltWinters
from forecast_methods.Naive import NaivePrediction
from forecast_methods.Arima import ARIMA
from forecast_methods.LSTM import LSTM, data_to_windowed_data, windowed_df_to_d_x_y
from forecast_methods.LSTM_OS import LSTM_OS

"""
Datei zur Evaluierung der Performance der einzelnen Prognosemethoden.
Außerdem wurde hiermit die Analyse der Parametereinstellung für Holt-Winters und ARIMA durchgeführt.
Dazu können Parameter in den einzelnen Funktionen angepasst werden.
Der zeitliche Horizont wird in dieser Datei angepasst.
Es wurden auch weitere Analysen hiermit gefahren, dazu wurde das Coding jeweils angepasst.
"""

def MAE(y_true, y_pred):
    """
    Berechnet "Mean Absolute Error" skaliert
    :param y_true, y_pred: tatsächliche und prognostizierte Kurse
    :return mea_scaled (Mean Absolute Error skaliert)
    """

    mae = mean_absolute_error(y_true, y_pred)
    average = np.mean(y_true)
    mae_scaled = round(mae/average, 4)
    return mae_scaled

stocks = ["ALV.DE", "DPW.DE", "AMZ.DE", "MDO.DE", "NVD.DE", "^MDAXI"] #Ausgewählte Aktien zum Vergleich
end_zeitpunkte = ["2023-02-18", "2023-05-17", "2023-07-07", "2022-12-16"] #Ausgewählte Zeiträume zum Vergleich
end_zeitpunkte_ts = [datetime.strptime(end_zeitpunkte[0], "%Y-%m-%d").date(), #Wandle Datum in Date-Format um
                      datetime.strptime(end_zeitpunkte[1], "%Y-%m-%d").date(),
                      datetime.strptime(end_zeitpunkte[2], "%Y-%m-%d").date(),
                      datetime.strptime(end_zeitpunkte[3], "%Y-%m-%d").date()]

#Zeitraum der historischen Daten anpassen: HW (5 Jahre), ARIMA(2 Jahre), LSTM-Modelle (3 Jahre) oder manuell für Analyse HW/ARIMA
time_horizont = 3
start_zeitpunkte_ts = [end_zeitpunkte_ts[0] - relativedelta(years = time_horizont), #Bestimme Start für die historischen Daten
                        end_zeitpunkte_ts[1] - relativedelta(years = time_horizont),
                        end_zeitpunkte_ts[2] - relativedelta(years = time_horizont),
                        end_zeitpunkte_ts[3] - relativedelta(years  =time_horizont)]

counter = 0 #Zur Erstellung des DF
for stock in stocks: #Iteriere über alle Aktien
    data = yf.download(stock, period = "max") #Downloade historische Daten
    data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True) #Entferne nicht benötigte Spalten
    for i in range(len(end_zeitpunkte)): #Iteriere über alle Zeiträume
        print(stock,i)
        filtered_df = data.loc[start_zeitpunkte_ts[i]: end_zeitpunkte_ts[i]] #Filtere Datensatz bezüglich des Zeitraums
        hist_data = filtered_df[: len(filtered_df) - 10] #Historische Daten für Prognose
        hist_data_for_function = hist_data.copy() #Für den Funktionsaufruf, da hist_data bei LSTM Methoden verändert wird
        prog_data = filtered_df[len(filtered_df) - 10: ] #Tatsächliche Kurse der voherzusagenden Tage
        index_list = prog_data.index.tolist() #Speichere Daten in Liste

        forecast_df = NaivePrediction(hist_data_for_function) #Funktionsaufruf der Methode; Möglichkeiten: Holt-Winters, ARIMA, LSTM, LSTM_OS, NaivePrediction

        forecast_df.set_index(pd.Index(index_list), inplace = True) #Verändere Index zu Datum
        result = pd.concat([prog_data, forecast_df], axis = 1) #Füge DF zusammen (untereinander)
        #Berechne Kennzahlen
        one_day = MAE([result["Close"][0]], [result["Predicted Price"][0]])
        three_day = MAE(result["Close"][:3], result["Predicted Price"][:3])
        five_day = MAE(result["Close"][:5], result["Predicted Price"][:5])
        ten_day = MAE(result["Close"], result["Predicted Price"])
        if counter == 0: #Falls 1. Run: Erstelle DF
            df_full = pd.DataFrame({counter: [one_day,three_day,five_day,ten_day]})
        else:
            df_new = pd.DataFrame({counter: [one_day,three_day,five_day,ten_day]})
            df_full = df_full.join(df_new) #Füge DF zusammen (nebeneinander)
        counter = counter + 1 #Erhöhe Counter, damit kein weiterer DF neu angelegt wird

average = df_full.mean(axis = 1) * 100 #Berechne Mittelwerte der einzelnen Kennzahlen
average_df = pd.DataFrame(average, columns = ["average MPAE"]) #Erstelle DF
average_df.loc[4] = average_df["average MPAE"].mean() #Berechne Durchschnitt der Kennzahlen
new_index = ["1Tag", "3Tage", "5Tage", "10Tage", "Average"] 
average_df.set_index(pd.Index(new_index), inplace = True) #Ändere Index
print(average_df)
#average_df.to_csv("solution_Naive.csv",sep = ";") #Speichere Ergebnisse im DF ab
