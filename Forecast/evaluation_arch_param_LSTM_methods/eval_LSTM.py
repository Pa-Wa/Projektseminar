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
Datei um die beste Architektur und Parametereinstellungen des LSTM Modells zu finden (anhand RMSE%).
Diese Datei kann durch leichte Anpassungen für alle analysierten Elemente genutzt werden (momentane Einstellung: Architektur-Analyse)
Dazu muss manuell die LSTM Funktion angepasst werden (deshlab ist diese in der Datei inkludiert).
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

stocks = ["ALV.DE", "AMZ.DE", "MDO.DE"] #Ausgewählte Aktien
test_runs = 5 #Mehrere Runs, damit bessere Vergleichbarkeit der Ergebnisse vorliegt
neurons = [32, 64, 128] #Verschiedene Neuronen je LSTM-Schicht
for neuron in neurons: #Iterieret Über Neuronen
    counter_df = 0 #Zur DF Erstellung
    for stock in stocks: #Iteriert über Aktien
        hist_data = yf.download(stock, start = "2020-06-23", end = "2023-06-23") #Downloaded historische Daten
        hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True) #Entfernt unnötige Spalten
        data_for_scaler = hist_data.copy() #Kopie, da Skalieren den DF ändert
        scaler = MinMaxScaler(feature_range = (0, 1)) #Definiert Scaler
        data_for_scaler[["Close"]] = scaler.fit_transform(data_for_scaler[["Close"]]) #Skaliert Daten
        data_for_func = data_for_scaler.copy() #Kopie, da Windowing den DF verändert
        window_size = 40 #Definiert Windowgröße
        windowed_df = data_to_windowed_data(data_for_func, window_size) #Windowing Technik
        dates, X, y = windowed_df_to_d_x_y(windowed_df)
        
        train_split = int(len(dates) * 0.8) #Aufteilen der Daten in Training und Validierung
        dates_train, X_train, y_train = dates[:train_split], X[:train_split], y[:train_split]
        dates_val, X_val, y_val = dates[train_split:], X[train_split:], y[train_split:]
        
        for runs in range(test_runs): #Mehrere Durchläufe
            print(neuron, stock, runs)
            
            train_predictions, val_predictions = LSTM(X_train, y_train, X_val, y_val, window_size, neuron) #Methodenaufruf

            train_predictions = scaler.inverse_transform(train_predictions) #Zurück Skalieren
            val_predictions = scaler.inverse_transform(val_predictions)
            y_train_rescaled = y_train.reshape(len(y_train), 1) #Zurück zum Ausgangsformat
            y_train_rescaled = scaler.inverse_transform(y_train_rescaled) #Zurück Skalieret
            average_price_train = y_train_rescaled.mean() #Berechnet Durchschnittskurs im Training
            y_val_rescaled = y_val.reshape(len(y_val), 1)
            y_val_rescaled = scaler.inverse_transform(y_val_rescaled)
            average_price_val = y_val_rescaled.mean()
            train_rmse = math.sqrt(mean_squared_error(y_trained, train_predictions)) #Berechnet RMSE
            val_rmse = math.sqrt(mean_squared_error(y_valed, val_predictions))

            if runs == 0: #Falls 1. Durchlauf
                df_RMSE = pd.DataFrame({"RMSE_Train": train_rmse, "RMSE_Valid": val_rmse}, index = [0]) #Erstellt DF
            else: 
                df_RMSE.loc[runs] = [train_rmse, val_rmse] #Fügt Zeile zum DF hinzu
            if runs == test_runs-1: #Falls letzter Durchlauf einer Aktie
                train_rmse_average = df_RMSE["RMSE_Train"].mean() #Berechnet Mittelwert des RMSE der Durchläufe
                val_rmse_average = df_RMSE["RMSE_Valid"].mean()
                train_rmse_average_procent = train_rmse_average/average_price_train #Berechnet RMSE% der Aktie
                val_rmse_average_procent = val_rmse_average/average_price_val
                if counter_df == 0: #Falls 1. Run einer Parametereinstellung
                    df_RMSE_full = pd.DataFrame({"RMSE_Train": train_rmse_average_procent, "RMSE_Valid": val_rmse_average_procent}, index = [0]) #Erstellt DF
                else:
                    df_RMSE_full.loc[counter_df] = [train_rmse_average_procent, val_rmse_average_procent] #Fügt Zeile zum DF hinzu
        counter_df = counter_df + 1 #Erhöhe Counter
    average_rmse_p_train =  df_RMSE_full["RMSE_Train"].mean() #Bildet Mittelwert aus allen Aktien-Performances fürs Training
    average_rmse_p_valid =  df_RMSE_full["RMSE_Valid"].mean()
    MSE_df = pd.DataFrame({"RMSE_Train_P": average_rmse_p_train, "RMSE_Valid_P": average_rmse_p_valid}, index= [0]) #Erstellt End-DF
    #MSE_df.to_csv(f"solution{neuron}.csv",sep = ";") #Speichert DF
