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

aktien = ["ALV.DE", "DPW.DE", "AMZ.DE", "MDO.DE", "NVD.DE", "^MDAXI"]
end_zeitpunkte = ["2023-02-18", "2023-05-17", "2023-07-07", "2022-12-16"]
end_zeitpunkte_ts = [datetime.strptime(end_zeitpunkte[0], "%Y-%m-%d").date(),
                      datetime.strptime(end_zeitpunkte[1], "%Y-%m-%d").date(),
                      datetime.strptime(end_zeitpunkte[2], "%Y-%m-%d").date(),
                      datetime.strptime(end_zeitpunkte[3], "%Y-%m-%d").date()]

#Zeitraum der historischen Daten anpassen: HW (5 Jahre), ARIMA(2 Jahre), LSTM-Modelle (3 Jahre)
time_horizont = 3
start_zeitpunkte_ts = [end_zeitpunkte_ts[0]-relativedelta(years=time_horizont),
                        end_zeitpunkte_ts[1]-relativedelta(years=time_horizont),
                        end_zeitpunkte_ts[2]-relativedelta(years=time_horizont),
                        end_zeitpunkte_ts[3]-relativedelta(years=time_horizont)]

counter = 0
for aktie in aktien:
    data = yf.download(aktie, period = "max")
    data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
    for i in range(len(end_zeitpunkte)):
        print(aktie,i)
        filtered_df = data.loc[start_zeitpunkte_ts[i]: end_zeitpunkte_ts[i]]
        hist_data = filtered_df[: len(filtered_df) - 10]
        hist_data_for_function = hist_data.copy()
        prog_data = filtered_df[len(filtered_df) - 10: ]
        index_list = prog_data.index.tolist()

        forecast_df = NaivePrediction(hist_data_for_function)

        forecast_df.set_index(pd.Index(index_list), inplace = True)
        result = pd.concat([prog_data, forecast_df], axis = 1)
        one_day = MAE([result["Close"][0]], [result["Predicted Price"][0]])
        three_day = MAE(result["Close"][:3], result["Predicted Price"][:3])
        five_day = MAE(result["Close"][:5], result["Predicted Price"][:5])
        ten_day = MAE(result["Close"], result["Predicted Price"])
        if counter == 0:
            df_full = pd.DataFrame({counter: [one_day,three_day,five_day,ten_day]})
        else:
            df_new = pd.DataFrame({counter: [one_day,three_day,five_day,ten_day]})
            df_full = df_full.join(df_new)
        counter = counter + 1

mittelwerte = df_full.mean(axis = 1) * 100
mittelwerte_df = pd.DataFrame(mittelwerte, columns = ["average MPAE"])
mittelwerte_df.loc[4] = mittelwerte_df["average MPAE"].mean()
neuer_index = ["1Tag", "3Tage", "5Tage", "10Tage", "Average"]
mittelwerte_df.set_index(pd.Index(neuer_index), inplace = True)
print(mittelwerte_df)
#mittelwerte_df.to_csv("solution_Arima_20Jahr.csv",sep = ";")
