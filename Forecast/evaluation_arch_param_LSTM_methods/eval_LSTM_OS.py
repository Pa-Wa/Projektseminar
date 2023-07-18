import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from other_func.performance_measurement_LSTM_OSpy import evaluate_model

"""
Datei um die beste Architektur und Parametereinstellungen des LSTM One-Shot Modells zu finden (anhand RMSE%).
Diese Datei kann durch leichte Anpassungen für alle analysierten Elemente genutzt werden (momentane Einstellung: Architektur-Analyse)
Dazu muss manuell die LSTM Funktion angepasst werden (deshlab ist diese in der Datei inkludiert).
"""

def LSTM_OS(x_train, Y_train, x_val, Y_val, window_size, units, prediction_sz):
    """
    LSTM One-Shot Modell wird trainiert und im Anschluss werden die Trainings- und Validierungsdaten vorhergesagt
    :param x_train, Y_train, x_val, Y_val: Trainings- und Validierungsdaten
    :param window_size: Windowgröße
    :param units: Anzahl der Neuronen je LSTM Schicht
    :param prediction_sz: Anz. vorherzusagender Tage
    :return: train_predict, val_predict (Trainings- und Validierungsvorhersagen)
    """

    model = Sequential([layers.Input((window_size, 1)), #LSTM-OS-Architektur
                    layers.LSTM(units, return_sequences = True),
                    layers.LSTM(units, return_sequences = True),
                    layers.LSTM(units, return_sequences = False),
                    layers.Dense(32),
                    layers.Dense(prediction_sz)])

    model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"]) #Definiert Performance Setup
    
    checkpoint_filepath = "best_model.h5"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_loss", mode = "min",
                                                                save_best_only = True) #Speichere bestes Modell ab

    model.fit(x_train, Y_train, validation_data = (x_val, Y_val), epochs = 80, verbose = 0, callbacks = [model_checkpoint_callback], batch_size = 32) #Trainiert Modell
    
    model = load_model(checkpoint_filepath) #Lädt bestes Modell

    train_predict = model.predict(x_train) #Vorhersage der Trainigs- und Validierungsdaten
    val_predict = model.predict(x_val)

    return train_predict, val_predict


stocks = ["ALV.DE","AMZ.DE","MDO.DE"] #Ausgewählte Aktien
test_runs = 5 #Mehrere Durchläufe, um Vergleichbarkeit zu gewährleisten
neurons = [32, 64, 128] #Variiert Anz. der Neuronen je LSTM Schicht
for neuron in neurons: #Iteriert über Neuronen-Anzahl
    counter_df = 0 #Zur Erstellung der DF
    for stock in stocks: #Iteriert über Aktien
        data = yf.download(stock, start = "2020-06-23", end = "2023-06-23") #Downloaded historische Daten
        data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True) #Entfernt unnütze Spalten
        scaler = MinMaxScaler(feature_range = (0, 1)) #Definiert Scaler
        data_scaled = scaler.fit_transform(data) #Skaliert Daten
        data_array = np.array(data_scaled)
        X, y = [], []
        window_size = 20 #Definiert Windowgröße
        prediction_size = 10 #Definiert Anz. vorherzusagender Tage
        for i in range(window_size, len(data_scaled)-prediction_size+1): #Windowing-Technik
            X.append(data_scaled[i-window_size:i])
            y.append(data_scaled[i:i+prediction_size])
        X, y = np.array(X), np.array(y)
        train_split = int(len(X) * 0.8) #Aufteilen der Daten in Training und Validierung
        X_train = X[:train_split]
        y_train = y[:train_split]
        X_val = X[train_split:] 
        y_val = y[train_split:]
        for runs in range(test_runs): #Mehrere Durchläufe mit gleichem Setting
            print(neuron, stock, runs)
            train_predictions, val_predictions = LSTM_OS(X_train, y_train, X_val, y_val, window_size, neuron, prediction_size) #Funktionsaufruf
            train_predictions = scaler.inverse_transform(train_predictions) #Zurückskalieren
            val_predictions = scaler.inverse_transform(val_predictions)
            y_true_shaped_train = y_train.reshape(len(y_train), prediction_size) #Zurück ins Ausgangsformat
            y_true_shaped_val = y_val.reshape(len(y_val), prediction_size)
            y_true_rescaled_train = scaler.inverse_transform(y_true_shaped_train) #Zurückskalieren
            y_true_rescaled_val = scaler.inverse_transform(y_true_shaped_val)
            total_mse_train, average_price_train = evaluate_model(y_true_rescaled_train, train_predictions) #Berechne RMSE
            total_mse_val, average_price_val = evaluate_model(y_true_rescaled_val, val_predictions)
            if runs == 0: #Falls 1. Durchgang, dann erstelle neuen DF
                df_RMSE = pd.DataFrame({"RMSE_Train": total_mse_train, "RMSE_Valid": total_mse_val}, index = [0])
            else: 
                df_RMSE.loc[runs] = [total_mse_train, total_mse_val] #Füge Zeile zum bestehenden DF hinzu
            if runs == test_runs - 1: #Falls letzter Durchlauf einer Aktie
                train_rmse_average = df_RMSE["RMSE_Train"].mean() #Berechne Mittelwert des RMSE der Runs einer Aktie
                val_rmse_average = df_RMSE["RMSE_Valid"].mean()
                train_rmse_average_procent = train_rmse_average/average_price_train #Berechne RMSE% einer Aktie
                val_rmse_average_procent = val_rmse_average/average_price_val
                if counter_df == 0: #Falls 1. Iteration überhaupt, erstelle neuen DF
                    df_RMSE_full = pd.DataFrame({"RMSE_Train": train_rmse_average_procent, "RMSE_Valid": val_rmse_average_procent}, index = [0])
                else:
                    df_RMSE_full.loc[counter_df] = [train_rmse_average_procent, val_rmse_average_procent] #Füge Zeile zum bestehenden DF hinzu
        counter_df = counter_df + 1 #Erhöhe Counter, damit kein neuer DF erstellt wird im momentanen Setting

    average_rmse_p_train = df_RMSE_full["RMSE_Train"].mean() #Bilde Mittelwert des RMSE% über alle Aktien
    average_rmse_p_valid = df_RMSE_full["RMSE_Valid"].mean()
    MSE_df = pd.DataFrame({"RMSE_Train_P": average_rmse_p_train, "RMSE_Valid_P": average_rmse_p_valid}, index = [0]) #Erstelle End - DF
    #MSE_df.to_csv(f"solution{neuron}.csv",sep = ";") #Speichere DF
