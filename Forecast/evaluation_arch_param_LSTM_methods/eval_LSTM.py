import yfinance as yf
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from other_func.windowing import data_to_windowed_data, windowed_df_to_d_x_y

"""
Datei um die beste Architektur und Parametereinstellungen des LSTM Modells zu finden.
Diese Datei kann durch leichte Anpassungen für alle analysierten Elemente genutzt werden (momentane Einstellung: Architektur-Analyse)
Dazu muss manuell die LSTM Funktion angepasst werden (deshlab ist diese in der Datei inkludiert)
"""

def LSTM(x_train, Y_train, x_val, Y_val, window, units):
    """
    LSTM Modell wird trainiert und im Anschluss werden die Trainings- und Validierungsdaten vorhergesagt
    :param x_train, Y_train, x_val, Y_val: Trainings- und Validierungsdaten
    :param window: Windowgröße
    :param units: Anzahl der Neuronen je LSTM Schicht
    :return: train_predict, val_predict (Trainings- und Validierungsvorhersagen)
    """

    model = Sequential([layers.Input((window, 1)), #Architektur des LSTM
                        layers.LSTM(units, return_sequences = True),
                        layers.LSTM(units, return_sequences = True),
                        layers.LSTM(units, return_sequences = False),
                        layers.Dense(32),
                        layers.Dense(1)])
    model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"]) #Definiert Performance-Messung
    
    checkpoint_filepath = "best_model.h5" 
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_loss", mode = "min",
                                                                save_best_only = True) #Speichere bestes Modell ab

    model.fit(x_train, Y_train, validation_data = (x_val, Y_val), epochs = 50, verbose = 0, batch_size = 32, callbacks = [model_checkpoint_callback]) #Trainiert Modell und wertet anhand Vald-Set aus

    model = load_model(checkpoint_filepath) #Lädt bestes Modell

    train_predict = model.predict(x_train) #Vorhersage der Trainigs- und Validierungsdaten
    val_predict = model.predict(x_val)

    return train_predict, val_predict

stocks = ["ALV.DE", "AMZ.DE", "MDO.DE"]
test_runs = 5
counter = 0
neurons = [32, 64, 128]
for neuron in neurons:
    counter_df = 0
    for stock in stocks:
        hist_data = yf.download(stock, start = "2020-06-23", end = "2023-06-23")
        hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
        data = hist_data[: len(hist_data)-10]
        data_for_scaler = data.copy()
        prog_data = hist_data[len(hist_data)-10 :]
        index_list = prog_data.index.tolist()
        scaler = MinMaxScaler(feature_range = (0, 1))
        data_for_scaler[["Close"]] = scaler.fit_transform(data_for_scaler[["Close"]])
        data_for_func = data_for_scaler.copy()
        window_size = 40
        windowed_df = data_to_windowed_data(data_for_func, window_size)
    

        dates, X, y = windowed_df_to_d_x_y(windowed_df)
        train_split = int(len(dates) * 0.8)

        dates_train, X_train, y_train = dates[:train_split], X[:train_split], y[:train_split]
        dates_val, X_val, y_val = dates[train_split:], X[train_split:], y[train_split:]
        for runs in range(test_runs):
            print(neuron, stock, runs)
            train_predictions, val_predictions = LSTM(X_train, y_train, X_val, y_val, window_size, neuron)

            train_predictions = scaler.inverse_transform(train_predictions)
            val_predictions = scaler.inverse_transform(val_predictions)

            y_trained = y_train.reshape(len(y_train), 1)
            y_trained = scaler.inverse_transform(y_trained)
            average_price_train = y_trained.mean()
            y_valed = y_val.reshape(len(y_val), 1)
            y_valed = scaler.inverse_transform(y_valed)
            average_price_val = y_valed.mean()
            train_rmse = math.sqrt(mean_squared_error(y_trained, train_predictions))
            val_rmse = math.sqrt(mean_squared_error(y_valed, val_predictions))

            if runs == 0:
                df_RMSE = pd.DataFrame({"RMSE_Train": train_rmse, "RMSE_Valid": val_rmse}, index = [0])
            else: 
                df_RMSE.loc[runs] = [train_rmse, val_rmse] 
            if runs == test_runs-1:
                train_rmse_average = df_RMSE["RMSE_Train"].mean()
                val_rmse_average = df_RMSE["RMSE_Valid"].mean()
                train_rmse_average_procent = train_rmse_average/average_price_train
                val_rmse_average_procent = val_rmse_average/average_price_val
                if counter_df == 0:
                    df_RMSE_full = pd.DataFrame({"RMSE_Train": train_rmse_average_procent, "RMSE_Valid": val_rmse_average_procent}, index = [0])
                else:
                    df_RMSE_full.loc[counter_df] = [train_rmse_average_procent, val_rmse_average_procent]
        counter_df = counter_df + 1
    counter=counter+1
    average_rmse_p_train =  df_RMSE_full["RMSE_Train"].mean()   
    average_rmse_p_valid =  df_RMSE_full["RMSE_Valid"].mean()
    MSE_df = pd.DataFrame({"RMSE_Train_P": average_rmse_p_train, "RMSE_Valid_P": average_rmse_p_valid}, index= [0])
    #MSE_df.to_csv(f"solution{neuron}.csv",sep = ";")
