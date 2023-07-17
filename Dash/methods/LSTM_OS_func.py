import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def evaluate_model(y_true, y_predicted):
    """
    Evaluiert die Performance des Modells auf den Trainings- und Validierungsdaten
    :param y_true, y_predicted: Tatsächliche Kurse und vorhergesagte Kurse
    :return total_score (MSE), total_mean_price (Durschnittskurs), total_mae_score (MAE)
    """
    
    total_score = 0
    mean_price = 0
    abs_dif = 0
    #Berechne Kennzahlen:
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]): #Gehe jedes Element durch und berechne dafür die Kennzahl
            mean_price = mean_price + y_true[row, col]
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
            abs_dif = abs_dif + (abs(y_true[row, col] - y_predicted[row, col]))
    total_score = np.sqrt(total_score/(y_true.shape[0] * y_predicted.shape[1])) #Berechne Durchschnitt
    total_mae_score = abs_dif/(y_true.shape[0] * y_predicted.shape[1])
    total_mean_price = mean_price/(y_true.shape[0] * y_predicted.shape[1])
    return total_score, total_mean_price, total_mae_score

def LSTM_OS(x_train, Y_train, x_val, Y_val, window, prediction_sz, last_prices, scaler, token, days):
    """
    Führt anhand der LSTM One-Shot Methode eine Kursprognose aus
    :param x_train, Y_train, x_val, Y_val: Input/Output des Trainings-/Validierungssets
    :param window: Windowgröße
    :param prediction_sz: Anz. vorherzusagender Tage
    :param last_prices: Letzte Preise je Windowgröße
    :param scaler: Scaler
    :param token: Aktien Token
    :param days: Datum
    :return: train_predict, val_predict (prognostizierte Trainings-/ Validierungsdaten)
    :return: forecast_df (DF der Vorhersage)
    """

    #Vortrainierte Modelle für Tesla und Nvidia (nicht UpToDate, Datengrundlage: 10.07.2023)
    if token == "TSLA" and days == 10:
        model = tf.keras.models.load_model("saved_model/LSTM_OS/TSLA")
    elif token == "NVDA" and days == 10:
        model = tf.keras.models.load_model("saved_model/LSTM_OS/NVIDIA")
        
    else:
        model = Sequential([layers.Input((window, 1)), #Architektur des LSTM
                        layers.LSTM(32),
                        layers.Dense(prediction_sz)])

        model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"]) #Definiert Performance Setup
        model.fit(x_train, Y_train, validation_data = (x_val, Y_val), epochs = 50, verbose = 0, batch_size = 32) #Trainiert Modell

    train_predict = model.predict(x_train) #Prognostiziert Trainings/-Vald.daten
    val_predict = model.predict(x_val)
    
    #Prognose
    last_prices_array = np.array(last_prices)
    input_pred = np.expand_dims(last_prices_array, axis = 0) #Dimension erhöhen
    predictions = model.predict(input_pred) #Vorhersage
    predictions = scaler.inverse_transform(predictions) #Zurück skalieren
    array_predictions = [value for sublist in predictions for value in sublist]
    forecast_df = pd.DataFrame(array_predictions, columns=["Close"]) #Vorhersage im DF abspeichern
    return train_predict, val_predict, forecast_df

def LSTM_OS_man_pred(history_data, token, date):
    """
    Führt anhand des LSTM One-Shot Modells eine Vorhersage durch
    :param history_data: Historische Daten
    :param token: Aktien Token
    :param date: Startdatum für die Prognose
    :return: forecast_df (DF der Vorhersage
    """
    
    scaler = MinMaxScaler(feature_range = (0, 1)) #Definiere Scaler
    data_scaled = scaler.fit_transform(history_data) #Skaliere
    X, y = [], []
    window_size = 30 #Definiert Windowgröße und Anz. zu prognostizierender Tage
    prediction_size = 10
    for i in range(window_size, len(data_scaled)- prediction_size + 1): #Wende Windowing-Technik an, erzeuge Input/Output
        X.append(data_scaled[i-window_size: i])
        y.append(data_scaled[i: i+prediction_size])
    X, y = np.array(X), np.array(y)
    train_split = int(len(X) * 0.8) #Teilt Daten in Training und Validierung auf
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:] 
    y_val = y[train_split:]

    #Vortrainierte Modell (Datengrundlage: 23.06.2023)
    if token == "TSLA" and date == "2023-06-24":
        model = tf.keras.models.load_model("saved_model/LSTM_OS/TSLA_man")
    elif token == "NVDA" and date == "2023-06-24":
        model = tf.keras.models.load_model("saved_model/LSTM_OS/NVIDIA_man")
    else: #Identisches Vorgehen wie oben
        model = Sequential([layers.Input((window_size, 1)),
                        layers.LSTM(32),
                        layers.Dense(prediction_size)])
        model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics=["mean_absolute_error"])
        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, verbose = 0, batch_size = 32)

    last_prices = data_scaled[-1*window_size:]
    last_prices_array = np.array(last_prices)
    input_pred = np.expand_dims(last_prices_array, axis = 0)
    predictions = model.predict(input_pred)
    predictions = scaler.inverse_transform(predictions)
    array_predictions = [value for sublist in predictions for value in sublist]
    forecast_df = pd.DataFrame(array_predictions, columns = ["Close"])
    return forecast_df
