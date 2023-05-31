import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import pandas_market_calendars as mcal

#Datenabruf:
ticker = yf.Ticker("TSLA")
hist_data = ticker.history(period = "max")

#Vorarbeit:
hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis = 1, inplace = True)
hist_data["Niveau"] = np.nan
hist_data["Trend"] = np.nan
hist_data["Prognose"] = np.nan
hist_data = hist_data.reset_index()
hist_data['Date'] = hist_data['Date'].dt.date

#Einstellungen:
alpha = 0.2
betha = 0.2
vorhersage = 10
nyse = mcal.get_calendar('NYSE')

#Initialisierung:
for i in range(len(hist_data)):
    if i == 0: #Initialisierung
        hist_data.loc[i, "Niveau"] = hist_data.loc[i, "Close"]
        hist_data.loc[i, "Trend"] = 0
        hist_data.loc[i, "Prognose"] = hist_data.loc[i, "Close"]
    else:
        hist_data.loc[i, "Prognose"] = hist_data.loc[i-1, "Niveau"] + hist_data.loc[i-1, "Trend"]
        hist_data.loc[i, "Niveau"] = alpha * hist_data.loc[i, "Close"] + (1-alpha) * (hist_data.loc[i-1, "Niveau"] + hist_data.loc[i-1, "Trend"])
        hist_data.loc[i, "Trend"] = betha * (hist_data.loc[i, "Niveau"] - hist_data.loc[i-1, "Niveau"]) + (1-betha) * hist_data.loc[i-1, "Trend"]

#Prognose:
last_row= len(hist_data) - 1
last_date = hist_data.loc[last_row, "Date"]
next_possible_dates = nyse.valid_days(start_date = last_date, end_date = last_date + datetime.timedelta(days=30))
for ii in range(vorhersage):
    prognose = hist_data.loc[last_row, "Niveau"] + (ii+1) * hist_data.loc[last_row, "Trend"]
    next_possible_date = next_possible_dates[ii+1].date()
    dic = {'Date' : next_possible_date ,'Close' : np.nan, "Niveau": np.nan, "Trend": np.nan, 'Prognose' : prognose}
    data_frame = pd.DataFrame(dic, index=[0])
    hist_data = pd.concat([hist_data, data_frame], ignore_index = True)

print(hist_data)