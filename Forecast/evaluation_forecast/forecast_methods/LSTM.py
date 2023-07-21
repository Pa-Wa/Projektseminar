import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def LSTM(historical_data):
    """
    Führt mittel LSTM Methode eine Kursprognose aus
    :param historical_data: historische Daten 
    :return forecast_df (DF der Vorhersage)
    """

    scaler = MinMaxScaler(feature_range = (0, 1)) #Definiert Scler
    historical_data[["Close"]] = scaler.fit_transform(historical_data[["Close"]]) #Skaliert Daten
    window_size = 50 #Definiert Windowgröße
    windowed_df = data_to_windowed_data(historical_data, window_size) #Erzeugt Windowed DF
    dates, X, y = windowed_df_to_d_x_y(windowed_df)

    train_split = int(len(dates) * 0.8) #Split der Daten in Training und Validierung
    dates_train, X_train, y_train = dates[:train_split], X[:train_split], y[:train_split]
    dates_val, X_val, y_val = dates[train_split:], X[train_split:], y[train_split:]

    model = Sequential([layers.Input((window_size, 1)), #LSTM Architektur
                        layers.LSTM(64, return_sequences = False),
                        layers.Dense(32, activation = "relu"),
                        layers.Dense(1)])
    model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"]) #Definiere Performance-Messung
    
    checkpoint_filepath = "best_model.h5" 
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_loss", mode = "min",
                                                                save_best_only = True) #Speichere bestes Modell ab
    model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 80, verbose = 0, callbacks = [model_checkpoint_callback], batch_size = 32) #Trainiert Modell

    model = load_model(checkpoint_filepath) #Lädt bestes Modell

    #Vorhersage
    pre_last_window = X_val[-1] #Speichert letztes Window des Vald-Sets
    last_window = np.delete(pre_last_window, obj = [0])  #Löscht 1. Preis des Windows und fügt letzten Preis der historischen Daten (letztes Y des Vald-Sets) an
    
    last_price = historical_data.iloc[-1][-1]
    last_window = np.append(last_window, last_price)
    last_window = np.reshape(last_window, (len(last_window), 1))
    predictions = []
    for iii in range(10): #10 Vorhersagen
        next_prediction = model.predict(np.array([last_window])).flatten()
        next_window = np.delete(last_window, obj = [0]) #Nutzt vorhergesagten Wert für die nächste Vorhersage; passt Window demnach an
        next_window = np.append(next_window, next_prediction)
        last_window = np.reshape(next_window, (len(next_window), 1)) 
        real_prediction = scaler.inverse_transform(next_prediction.reshape(-1, 1)) #Zurück skalieren
        predictions.append(real_prediction)

    array_predictions = [predictions[ii][0][0] for ii in range(len(predictions))]
    forecast_df = pd.DataFrame(array_predictions, columns = ["Predicted Price"]) #Erstellt DF der Vorhersagen
    return forecast_df

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
