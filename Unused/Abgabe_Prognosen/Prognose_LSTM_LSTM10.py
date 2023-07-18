import yfinance as yf
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

"""
Verfahren, Architektur und Parameter wurden im Nachgang neu evaluiert!
"""

def LSTM10(hist_data):
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_scaled = scaler.fit_transform(hist_data)
    X, y = [], []
    window_size = 30
    prediction_size = 10
    for i in range(window_size, len(data_scaled)-prediction_size+1):
        X.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i:i+prediction_size])
    X_train, y_train = np.array(X), np.array(y)

    model = Sequential([layers.Input((window_size, 1)),
                    #layers.LSTM(units, return_sequences = True),
                    layers.LSTM(32),
                    layers.Dense(32, activation = "relu"),
                    layers.Dense(prediction_size)])

    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, mode='min')
    model.compile(loss = "mse", 
              optimizer=Adam(learning_rate = 0.001),
              metrics=["mean_absolute_error"])

    model.fit(X_train, y_train, epochs = 80, verbose = 0)
    #input_data = np.expand_dims(X_train[-1], axis=0)
    last_prices = data_scaled[-1*window_size:]
    last_prices_array = np.array(last_prices)
    input_pred = np.expand_dims(last_prices_array, axis=0)
    predictions = model.predict(input_pred)
    predictions = scaler.inverse_transform(predictions)
    array_predictions = [value for sublist in predictions for value in sublist]
    forecast_df = pd.DataFrame(array_predictions, columns=["Predicted Price"])
    return forecast_df

def data_to_windowed_data(df, windows):
    close_prices = df["Close"]
    for i in range(windows - 1, -1, -1):
        column_name = f"Target_{i+1}"
        df.loc[:, column_name] = close_prices.shift(i+1)
    df.reset_index(inplace = True)
    df.rename(columns = {"Date":"Target Date"}, inplace = True)
    col = df.pop("Close")
    df.insert(loc= len(df.columns) , column= "Target", value= col)
    df.dropna(inplace = True)
    return df
def windowed_df_to_d_x_y(wdf):
    df_as_np = wdf.to_numpy()
    dates = df_as_np[:,0]
    pre_matrix = df_as_np[:, 1:-1]
    X = pre_matrix.reshape((len(dates), pre_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)

def LSTM(hist_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    hist_data[["Close"]] = scaler.fit_transform(hist_data[["Close"]])

    window_size = 3
    windowed_df = data_to_windowed_data(hist_data, window_size)
        
    dates_train, X_train, y_train = windowed_df_to_d_x_y(windowed_df)
    model = Sequential([layers.Input((window_size, 1)),
                        layers.LSTM(32, return_sequences= False),
                        #layers.LSTM(units, return_sequences = True),
                        #layers.LSTM(units),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])

        #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, mode='min')

    model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.001),
                metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, epochs = 80, verbose = 0) # callbacks =  [early_stopping]

    pre_last_window = X_train[-1]
    last_window = np.delete(pre_last_window, obj = [0])
    last_price = hist_data.iloc[-1][-1]
    last_window = np.append(last_window, last_price)
    last_window = np.reshape(last_window, (len(last_window), 1))

    predictions = []
    for iii in range(10):
        next_prediction = model.predict(np.array([last_window])).flatten()
        next_window = np.delete(last_window, obj = [0])
        next_window = np.append(next_window, next_prediction)
        last_window = np.reshape(next_window, (len(next_window), 1))
        real_prediction = scaler.inverse_transform(next_prediction.reshape(-1, 1))
        predictions.append(real_prediction)

    array_predictions = [predictions[ii][0][0] for ii in range(len(predictions))]
    forecast_df = pd.DataFrame(array_predictions, columns=["Predicted Price"])
    return forecast_df

aktien = ["ALV.DE", "AMZ.DE", "DPW.DE", "MDO.DE", "NVD.DE", "^MDAXI"]#
liste=[]
for aktie in aktien:
    data = yf.download(aktie, period = "3y")
    data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
    forecast_df = LSTM10(data)
    print(forecast_df)
    forecast_list=forecast_df["Predicted Price"].tolist()
    day = 1
    #df = pd.DataFrame({"Gruppe(Nachname)": "Chevalaz-Wagener", "Aktie/Indize(Kürzel)": aktie, "Handelstag": 1, "Kurs": a}, index = [0])
    for i in forecast_list:
        a = round(i,2)
        b = ["Chevalaz-Wagener", aktie, day, a]
        liste.append(b)
        day = day+1
print(liste)
df = pd.DataFrame(liste, columns = ["Gruppe(Nachname)","Aktie/Indize(Kürzel)","Handelstag", "Kurs"])
df.to_csv("prediction_LSTM_OneShot.csv", sep = ";", index = False)
