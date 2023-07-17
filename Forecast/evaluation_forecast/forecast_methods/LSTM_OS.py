import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def LSTM_OS(historical_data):
    """
    Führt mittel LSTM One-Shot Methode eine Kursprognose aus
    :param historical_data: historische Daten 
    :return forecast_df (DF der Vorhersage)
    """

    scaler = MinMaxScaler(feature_range = (0, 1)) #Definiert Scaler
    data_scaled = scaler.fit_transform(historical_data) #Skaliert Daten
    X, y = [], []
    window_size = 30 #Definiert Windowgröße
    prediction_size = 10 #Definiert Anz. vorherzusagender Tage

    for i in range(window_size, len(data_scaled)-prediction_size+1): #Führt Windowing-Technik aus
        X.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i:i+prediction_size])
    X, y = np.array(X), np.array(y)

    train_split = int(len(X) * 0.8) #Split der Daten in Training und Validierung
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:] 
    y_val = y[train_split:] 

    model = Sequential([layers.Input((window_size, 1)), #LSTM-OS-Architektur
                    layers.LSTM(128, return_sequences = True),
                    layers.LSTM(128),
                    layers.Dense(prediction_size)])

    model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"]) #Definiert Performance Setup
    
    checkpoint_filepath = "best_model.h5"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_loss", mode = "min",
                                                                save_best_only = True) #Speichere bestes Modell ab

    model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 80, verbose = 0, callbacks = [model_checkpoint_callback], batch_size = 32) #Trainiert Modell
    
    model = load_model(checkpoint_filepath) #Lädt bestes Modell

    #Vorhersage
    last_prices = data_scaled[-1*window_size:] 
    last_prices_array = np.array(last_prices)
    input_pred = np.expand_dims(last_prices_array, axis=0) #Dimension erhöhen
    predictions = model.predict(input_pred) #Vorhersage
    predictions = scaler.inverse_transform(predictions) #Zurück skalieren
    array_predictions = [value for sublist in predictions for value in sublist]
    forecast_df = pd.DataFrame(array_predictions, columns=["Predicted Price"]) #Vorhersage im DF abspeichern
    return forecast_df
