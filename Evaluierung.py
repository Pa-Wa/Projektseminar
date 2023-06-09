import yfinance as yf
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
#Kann sein, dass du noch Pakete installieren musst, falls du sie noch nicht hast
#Inputs, die die Methode braucht einfügen!

def HoltWinter(historical_data):    
    model = ExponentialSmoothing(historical_data, trend='add', seasonal="add", seasonal_periods=7)
    model_fit = model.fit()
    model_fit = model.fit(smoothing_trend=0.2)
    forecast = model_fit.forecast(steps=10)
    forecast_df = pd.DataFrame(forecast, columns=["Predicted Price"])
    return forecast_df

def MEA(y_true, y_pred):
    mea = mean_absolute_error(y_true, y_pred)
    average = np.mean(y_true)
    mea_skaled = round(mea/average, 4)
    return mea_skaled


aktien = ["^GSPC","^RUT","NVD.DE","AMD.DE","TL0.DE","MDO.DE","SRB.DE","PFE.DE","U9R.F","1NBA.F"] 
end_zeitpunkte = ["2023-02-18","2023-05-17"]
end_zeitpunkte_ts = [datetime.strptime(end_zeitpunkte[0], "%Y-%m-%d").date(), datetime.strptime(end_zeitpunkte[1], "%Y-%m-%d").date()]

#hier Zeitraum der historischen Daten anpassen:
start_zeitpunkte_ts = [end_zeitpunkte_ts[0]-relativedelta(months=2), end_zeitpunkte_ts[1]-relativedelta(months=2)]

start_zeitpunkte = [start_zeitpunkte_ts[0].strftime("%Y-%m-%d"), start_zeitpunkte_ts[1].strftime("%Y-%m-%d")]
counter = 0
for i in range(len(end_zeitpunkte)):
    end_zeitpunkt = end_zeitpunkte[i]
    start_zeitpunkt = start_zeitpunkte[i]
    for aktie in aktien:
        data = yf.download(aktie, start= start_zeitpunkt, end = end_zeitpunkt)
        data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], axis = 1, inplace = True)
        hist_data = data[: len(data)-10]
        prog_data = data[len(data)-10 :]
        index_list = prog_data.index.tolist()
        
        #Methode einfügen! -> Wichtig enstehender DF muss wie besprochen aussehen und forecast_df heißen!
        forecast_df = HoltWinter(hist_data) #meine Methode


        forecast_df.set_index(pd.Index(index_list), inplace=True)
        result = pd.concat([prog_data, forecast_df], axis=1)
        one_day = MEA([result["Close"][0]], [result["Predicted Price"][0]])
        three_day = MEA(result["Close"][:3], result["Predicted Price"][:3])
        five_day = MEA(result["Close"][:5], result["Predicted Price"][:5])
        ten_day = MEA(result["Close"], result["Predicted Price"])
        if counter == 0:
            df_full = pd.DataFrame({counter: [one_day,three_day,five_day,ten_day]})    
        else:
            df_new = pd.DataFrame({counter: [one_day,three_day,five_day,ten_day]})
            df_full = df_full.join(df_new)
        counter = counter + 1

mittelwerte = df_full.mean(axis=1)*100
mittelwerte_df = pd.DataFrame(mittelwerte, columns=["average MPAE"])
mittelwerte_df.loc[4] = mittelwerte_df["average MPAE"].mean()
neuer_index = ["1Tag","3Tage","5Tage","10Tage","Average"]
mittelwerte_df.set_index(pd.Index(neuer_index),inplace=True)
print(mittelwerte_df)
mittelwerte_df.to_csv("solution.csv",sep = ";")

#für mich:
#Parameter einstellen und for Schleife
#Methode überprüfen Warning-Meldung