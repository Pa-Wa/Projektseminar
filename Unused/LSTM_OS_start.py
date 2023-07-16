import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

data = yf.download("NVDA", start="2020-06-22", end="2023-06-24")
data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
scaler = MinMaxScaler(feature_range = (0, 1))
data_scaled = scaler.fit_transform(data)
train_split = int(len(data_scaled) * 0.8)
data_array = np.array(data_scaled)
X, y = [], []
window_size = 30
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

model = Sequential([layers.Input((window_size, 1)),
                    layers.LSTM(128, return_sequences=True),
                    layers.LSTM(128),
                    layers.Dense(prediction_size)])

#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, mode='min')

model.compile(loss = "mse", 
              optimizer=Adam(learning_rate = 0.001),
              metrics=["mean_absolute_error"])

checkpoint_filepath = "model_saved.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    mode="min",
    save_best_only=True)


model.fit(X_train, y_train, validation_data = (X_val, y_val), callbacks = [model_checkpoint_callback], epochs = 80)
model = load_model(checkpoint_filepath)
#model.save('NVIDIA2')

last_prices = data_scaled[-1*window_size:]
last_prices_array = np.array(last_prices)
input_pred = np.expand_dims(last_prices_array, axis = 0)
predictions = model.predict(input_pred)
predictions = scaler.inverse_transform(predictions)
array_predictions = [value for sublist in predictions for value in sublist]
forecast_df = pd.DataFrame(array_predictions, columns = ["Close"])
print(forecast_df)
