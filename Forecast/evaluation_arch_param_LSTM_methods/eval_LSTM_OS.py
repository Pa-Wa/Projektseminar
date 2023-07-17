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


aktien = ["ALV.DE","AMZ.DE","MDO.DE"] #"DPW.DE",  "NVD.DE", "^MDAXI"

test_runs = 5
neurons = [32, 64, 128]
for neuron in neurons:
    counter_df = 0
    for aktie in aktien:
        data = yf.download(aktie, start = "2020-06-23", end = "2023-06-23") 
        data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
        scaler = MinMaxScaler(feature_range = (0, 1))
        data_scaled = scaler.fit_transform(data)
        data_array = np.array(data_scaled)
        X, y = [], []
        window_size = 20
        prediction_size = 10
        for i in range(window_size, len(data_scaled)-prediction_size+1):
            X.append(data_scaled[i-window_size:i])
            y.append(data_scaled[i:i+prediction_size])
        X, y = np.array(X), np.array(y)
        train_split = int(len(X) * 0.8)
        X_train = X[:train_split]
        y_train = y[:train_split]
        X_val = X[train_split:] 
        y_val = y[train_split:]
        for runs in range(test_runs):
            print(neuron, aktie, runs)
            train_predictions, val_predictions = LSTM_OS(X_train, y_train, X_val, y_val, window_size, neuron, prediction_size)
            train_predictions = scaler.inverse_transform(train_predictions)
            val_predictions = scaler.inverse_transform(val_predictions)
            y_true_shaped_train = y_train.reshape(len(y_train), prediction_size)
            y_true_shaped_val = y_val.reshape(len(y_val), prediction_size)
            y_true_train = scaler.inverse_transform(y_true_shaped_train)
            y_true_val = scaler.inverse_transform(y_true_shaped_val)
            total_mse_train, average_price_train = evaluate_model(y_true_train, train_predictions)
            total_mse_val, average_price_val = evaluate_model(y_true_val, val_predictions)
            if runs == 0:
                df_RMSE = pd.DataFrame({"RMSE_Train": total_mse_train, "RMSE_Valid": total_mse_val}, index = [0])
            else: 
                df_RMSE.loc[runs] = [total_mse_train, total_mse_val]
            if runs == test_runs - 1:
                train_rmse_average = df_RMSE["RMSE_Train"].mean()
                val_rmse_average = df_RMSE["RMSE_Valid"].mean()
                train_rmse_average_procent = train_rmse_average/average_price_train
                val_rmse_average_procent = val_rmse_average/average_price_val
                if counter_df == 0:
                    df_RMSE_full = pd.DataFrame({"RMSE_Train": train_rmse_average_procent, "RMSE_Valid": val_rmse_average_procent}, index = [0])
                else:
                    df_RMSE_full.loc[counter_df] = [train_rmse_average_procent, val_rmse_average_procent]
        counter_df = counter_df + 1

    average_rmse_p_train = df_RMSE_full["RMSE_Train"].mean()   
    average_rmse_p_valid = df_RMSE_full["RMSE_Valid"].mean()
    MSE_df = pd.DataFrame({"RMSE_Train_P": average_rmse_p_train, "RMSE_Valid_P": average_rmse_p_valid}, index = [0])
    #MSE_df.to_csv(f"solution{neuron}.csv",sep = ";")