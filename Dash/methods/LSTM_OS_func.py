import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def evaluate_model(y_true, y_predicted):
    total_score = 0
    mean_price = 0
    abs_dif = 0
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]):
            mean_price = mean_price + y_true[row, col]
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
            abs_dif = abs_dif + (abs(y_true[row, col] - y_predicted[row, col]))
    total_score = np.sqrt(total_score/(y_true.shape[0] * y_predicted.shape[1]))
    total_mae_score = abs_dif/(y_true.shape[0] * y_predicted.shape[1])
    total_mean_price = mean_price/(y_true.shape[0] * y_predicted.shape[1])
    return total_score, total_mean_price, total_mae_score

def LSTM_OS(x_train, Y_train, x_val, Y_val, window, prediction_sz, last_prices, scaler, token, days):
    if token == "TSLA" and days == 10:
        model = tf.keras.models.load_model("saved_model/LSTM_OS/TSLA")
    elif token == "NVDA" and days == 10:
        model = tf.keras.models.load_model("saved_model/LSTM_OS/NVIDIA")
    else:
        model = Sequential([layers.Input((window, 1)),
                        layers.LSTM(32),
                        layers.Dense(prediction_sz)])

        model.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ["mean_absolute_error"])
        model.fit(x_train, Y_train, validation_data = (x_val, Y_val), epochs = 25, verbose = 0, batch_size = 32)

    train_predict = model.predict(x_train)
    val_predict = model.predict(x_val)

    last_prices_array = np.array(last_prices)
    input_pred = np.expand_dims(last_prices_array, axis = 0)
    predictions = model.predict(input_pred)
    predictions = scaler.inverse_transform(predictions)
    array_predictions = [value for sublist in predictions for value in sublist]
    forecast_df = pd.DataFrame(array_predictions, columns=["Close"])
    return train_predict, val_predict, forecast_df

def LSTM_OS_man_pred(history_data, token, date):
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_scaled = scaler.fit_transform(history_data)
    X, y = [], []
    window_size = 30
    prediction_size = 10
    for i in range(window_size, len(data_scaled)-prediction_size+1):
        X.append(data_scaled[i-window_size: i])
        y.append(data_scaled[i: i+prediction_size])
    X, y = np.array(X), np.array(y)
    train_split = int(len(X) * 0.8)
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:] 
    y_val = y[train_split:]

    if token == "TSLA" and date == "2023-06-24":
        model = tf.keras.models.load_model("saved_model/LSTM_OS/TSLA_man")
    elif token == "NVDA" and date == "2023-06-24":
        model = tf.keras.models.load_model("saved_model/LSTM_OS/NVIDIA_man")
    else:
        model = Sequential([layers.Input((window_size, 1)),
                        layers.LSTM(32),
                        layers.Dense(prediction_size)])

        model.compile(loss = "mse", 
                optimizer=Adam(learning_rate = 0.001),
                metrics=["mean_absolute_error"])

        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 20, verbose = 0, batch_size = 32)

    last_prices = data_scaled[-1*window_size:]
    last_prices_array = np.array(last_prices)
    input_pred = np.expand_dims(last_prices_array, axis = 0)
    predictions = model.predict(input_pred)
    predictions = scaler.inverse_transform(predictions)
    array_predictions = [value for sublist in predictions for value in sublist]
    forecast_df = pd.DataFrame(array_predictions, columns = ["Close"])
    return forecast_df