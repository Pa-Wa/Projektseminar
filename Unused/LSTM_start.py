import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import tensorflow as tf

def data_to_windowed_data(df, windows):
    close_prices = df["Close"]
    for i in range(windows - 1, -1, -1):
        column_name = f"Target_{i+1}"
        df[column_name] = close_prices.shift(i+1)
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

data = yf.download("TSLA", start = "2020-06-23", end = "2023-06-23")
data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
MinMax = "full"
if MinMax == "full":
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[["Close"]]= scaler.fit_transform(data[["Close"]])
if MinMax == "split":
    train_split = int(len(data) * 0.8)
    train = data[:train_split]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train[["Close"]]= scaler.fit_transform(train[["Close"]])
    test = data[train_split:]
    test[["Close"]]= scaler.transform(test[["Close"]])
    data = pd.concat([train, test], axis=0)

window_size = 40
windowed_df = data_to_windowed_data(data, window_size)
dates, X, y = windowed_df_to_d_x_y(windowed_df)
train_split = int(len(dates) * 0.8)
dates_train, X_train, y_train = dates[:train_split], X[:train_split], y[:train_split]
dates_val, X_val, y_val = dates[train_split+window_size:], X[train_split+window_size:], y[train_split+window_size:]

model = Sequential([layers.Input((window_size, 1)),
                    #layers.LSTM(64, return_sequences = True),
                    layers.LSTM(32),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1)])

#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, mode='min') #Stoppt Modell sobald keine Besserung in 5 aufeinanderfolgenden Epochen

model.compile(loss="mse", 
              optimizer=Adam(learning_rate=0.001),
              metrics=["mean_absolute_error"])

checkpoint_filepath = "best_model.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    mode="min",
    save_best_only=True)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=32, callbacks = [model_checkpoint_callback]) # callbacks =  [early_stopping]

model = load_model(checkpoint_filepath)
train_predictions = model.predict(X_train)
train_predictions = scaler.inverse_transform(train_predictions)
val_predictions = model.predict(X_val)
val_predictions = scaler.inverse_transform(val_predictions)
y_train = y_train.reshape(len(y_train), 1)
y_train = scaler.inverse_transform(y_train)
y_val = y_val.reshape(len(y_val), 1)
y_val = scaler.inverse_transform(y_val)
train_rmse = math.sqrt(mean_squared_error(y_train, train_predictions))
val_rmse = math.sqrt(mean_squared_error(y_val, val_predictions))
val_mae = mean_absolute_error(y_val, val_predictions)
train_mae = mean_absolute_error(y_train, train_predictions)

print("Train RMSE:", train_mae)
print("Train RMSE:", train_rmse)
print("Validation MAE:", val_mae)
print("Validation RMSE:", val_rmse)

last_window = X_val[-1]
predictions = []
for i in range(10):
    next_prediction = model.predict(np.array([last_window])).flatten()
    next_window = np.delete(last_window, obj = [0])
    next_window = np.append(next_window, next_prediction)
    last_window = np.reshape(next_window, (len(next_window), 1))
    real_prediction = scaler.inverse_transform(next_prediction.reshape(-1, 1))
    predictions.append(real_prediction)

array = [predictions[i][0][0] for i in range(len(predictions))]
forecast_df = pd.DataFrame(array, columns=["Predicted Price"])
print(forecast_df)
