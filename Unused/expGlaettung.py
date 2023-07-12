import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import pandas_market_calendars as mcal

ticker = yf.Ticker("TSLA")
hist_data = ticker.history(period = "1mo")

hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis = 1, inplace = True)
hist_data["Prognose"] = np.nan
hist_data = hist_data.reset_index()
hist_data['Date'] = hist_data['Date'].dt.date

alpha = 0.3
vorhersage = 14
nyse = mcal.get_calendar('NYSE')

for i in range(len(hist_data)):
    if i == 0:
        hist_data.loc[i, "Prognose"] = hist_data.loc[i, "Close"]
    else:
        hist_data.loc[i, "Prognose"] = alpha * hist_data.loc[i-1, "Close"] + (1-alpha) * hist_data.loc[i-1, "Prognose"]

prognose = alpha * hist_data.loc[len(hist_data)-1, "Close"] + (1-alpha) * hist_data.loc[len(hist_data)-1, "Prognose"]
last_row= len(hist_data)-1

"""
Methode ohne Betrachtung von Börsentagen
for ii in range(vorhersage):
    dates = hist_data.loc[last_row, "Date"] + timedelta(days=1+ii)
    hist_data = hist_data.append({'Date' : dates , 'Close' : np.nan, 'Prognose' : prognose} , ignore_index=True)
"""

for ii in range(vorhersage):
    last_date = hist_data.loc[last_row+ii, "Date"]
    next_possible_dates = nyse.valid_days(start_date = last_date, end_date = last_date + datetime.timedelta(days=10))
    next_possible_date = next_possible_dates[1].date()
    hist_data = hist_data.append({'Date' : next_possible_date , 'Close' : np.nan, 'Prognose' : prognose} , ignore_index=True)

print(hist_data)
