import pandas as pd
import yfinance as yf
import numpy as np
from datetime import timedelta

ticker = yf.Ticker("TSLA")
hist_data = ticker.history(period = "1mo")

hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis = 1, inplace = True)
hist_data["Prognose"] = np.nan
hist_data = hist_data.reset_index()

alpha = 0.3
vorhersage = 14

for i in range(len(hist_data)):
    if i == 0:
        hist_data.loc[i, "Prognose"] = hist_data.loc[i, "Close"]
    else:
        hist_data.loc[i, "Prognose"] = alpha * hist_data.loc[i-1, "Close"] + (1-alpha) * hist_data.loc[i-1, "Prognose"]

prognose = alpha * hist_data.loc[len(hist_data)-1, "Close"] + (1-alpha) * hist_data.loc[len(hist_data)-1, "Prognose"]
last_row= len(hist_data)-1
print(last_row)
for ii in range(vorhersage):
    date = hist_data.loc[last_row, "Date"] + timedelta(days=1+ii)
    print(date)
    #hist_data.loc[len(hist_data)+ii] = [date, np.nan, prognose]
    hist_data = hist_data.append({'Date' : date , 'Close' : np.nan, 'Prognose' : prognose} , ignore_index=True)
print(hist_data)