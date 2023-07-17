import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def data_to_windowed_data(df, windows):
    """
    Wendet die Windowing-Technik an
    :param df: Dataframe der historischen Daten
    :param windows: Windowgröße
    :return: df (DF nach Anwendung der Windowing-Technik
    """
    
    close_prices = df["Close"]
    for i in range(windows - 1, -1, -1): #Geht alle Elemente des DF durch und erzeugt Windows
        column_name = f"Target_{i+1}"
        df.loc[:, column_name] = close_prices.shift(i+1) #Erzeugt Windows und speichert sie ab
    df.reset_index(inplace = True) #Index zu Spalte
    df.rename(columns = {"Date": "Target Date"}, inplace = True) #Umbenennung
    col = df.pop("Close") #Entferne Close-Spalte und füge sie mit anderem Namen an anderer Stelle ein
    df.insert(loc = len(df.columns), column = "Target", value = col) 
    df.dropna(inplace = True) #Entferne Nan's (erste Windowgröße-Elemente)
    return df

def windowed_df_to_d_x_y(wdf):
    """
    Umformatierung des Window-DF in bestimmtes Format für Tensorflow
    :param wdf: Windowed-DF
    :return: dates, X, Y (Daten, Input-Daten, Target-Daten)
    """
    
    df_as_np = wdf.to_numpy() #Zu numpy Format
    dates = df_as_np[:, 0] #Dates = 1.Spalte des DF
    pre_matrix = df_as_np[:, 1: -1] #Input-Daten (Window, ohne Target)
    X = pre_matrix.reshape((len(dates), pre_matrix.shape[1], 1)) #Umformatierung
    Y = df_as_np[:, -1] #Target-Daten
    return dates, X.astype(np.float32), Y.astype(np.float32)

def LSTM(x_train, Y_train, x_val, Y_val, window, last_price, scaler, token):
    """
    Erzeugt mittel LSTM-Methode Kursprognosen
    :param x_train, Y_train, x_val, Y_val: Trainings- und Validierungsdaten
    :param window: Windowgröße
    :param last_price: Letzter Preis der historischen Daten
    :param scaler: Min-Max Scaler
    :param token: Aktien Token
    :return: train_predict, val_predict (Trainings- und Validierungsvorhersagen)
    :return: forecast_df (DF der Kursprognosen
    """
    
    #Vortrainierte Modelle für TSLA und Nvidia (nicht UpToDate, Datengrundlage 10.07.2023)
    if token == "TSLA":
        model = tf.keras.models.load_model("saved_model/LSTM/TSLA")
    elif token == "NVDA":
        model = tf.keras.models.load_model("saved_model/LSTM/NVIDIA")
        
    else: #Falls nicht NVDA und TSLA, dann generiere Model (aufgrund der Ladezeiten ein sehr einfaches Modell)
        model = Sequential([layers.Input((window, 1)), #Architektur des LSTM
                            layers.LSTM(32, return_sequences = False),
                            layers.Dense(1)])
        model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"]) #Definiert Performance-Messung
        model.fit(x_train, Y_train, validation_data = (x_val, Y_val), epochs = 50, verbose = 0, batch_size = 32) #Trainiert Modell und wertet anhand Vald-Set aus

    train_predict = model.predict(x_train) #Vorhersage der Trainigs- und Validierungsdaten
    val_predict = model.predict(x_val)
    
    #Prognose
    pre_last_window = x_val[-1] #Speichert letztes Window des Vald-Sets
    last_window = np.delete(pre_last_window, obj = [0]) #Löscht 1. Preis des Windows und fügt letzten Preis der historischen Daten (letztes Y des Vald-Sets) an
    last_window = np.append(last_window, last_price) 
    last_window = np.reshape(last_window, (len(last_window), 1)) #Umformatierung
    predictions = []
    for iii in range(30): #Mache 30 Vorhersagen
        next_prediction = model.predict(np.array([last_window])).flatten() #Vorhersage
        next_window = np.delete(last_window, obj = [0]) #Nutzt vorhergesagten Wert für die nächste Vorhersage; passt Window demnach an
        next_window = np.append(next_window, next_prediction)
        last_window = np.reshape(next_window, (len(next_window), 1))
        real_prediction = scaler.inverse_transform(next_prediction.reshape(-1, 1)) #Wandelt vorhergesagten Kurs um
        predictions.append(real_prediction)
    array_predictions = [predictions[ii][0][0] for ii in range(len(predictions))]
    forecast_df = pd.DataFrame(array_predictions, columns=["Close"]) #Erstellt DF der Vorhersagen
    return train_predict, val_predict, forecast_df

def LSTM_man_pred(history_data, token, date):
    """
    Erzeugt Prognose anhand LSTM-Methode
    :param history_data: Historische Daten
    :param token: Aktien Token
    :param date: Datum ab dem die Prognose beginnen soll
    :return: forecast_df (DF der Vorhersage)
    """

    scaler = MinMaxScaler(feature_range = (0, 1)) #Definiere Scaler
    history_data[["Close"]] = scaler.fit_transform(history_data[["Close"]]) #Skaliere Daten
    window_size = 50 #Defniniere Windowgröße
    windowed_df = data_to_windowed_data(history_data, window_size) #Erstelle Windowed-DF   
    dates, X, y = windowed_df_to_d_x_y(windowed_df) #Erzeuge Input/Output 
    train_split = int(len(dates) * 0.8) 
    dates_train, X_train, y_train = dates[:train_split], X[:train_split], y[:train_split] #Trenne Daten in Trainings-Valdsets
    dates_val, X_val, y_val = dates[train_split:], X[train_split:], y[train_split:]

    #Vortrainierte Modelle für TSLA und Nvidia (nicht UpToDate, Datengrundlage 23.06.2023)
    if token == "TSLA" and date == "2023-06-24":
        model = tf.keras.models.load_model("saved_model/LSTM/TSLA_man")
    elif token == "NVDA" and date == "2023-06-24":
        model = tf.keras.models.load_model("saved_model/LSTM/NVIDIA_man")
        
    #Vergleiche Kommentierung von oben!
    else:
        model = Sequential([layers.Input((window_size, 1)), 
                            layers.LSTM(32, return_sequences= False),
                            layers.Dense(1)])

        model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"])
        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, verbose = 0, batch_size = 32)
    pre_last_window = X_val[-1]
    last_window = np.delete(pre_last_window, obj = [0])
    last_price = history_data.iloc[-1][-1]
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
    forecast_df = pd.DataFrame(array_predictions, columns = ["Close"])
    return forecast_df
