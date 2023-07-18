from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf
import pandas as pd

aktien = ["ALV.DE", "AMZ.DE", "DPW.DE", "MDO.DE", "NVD.DE", "^MDAXI"]
counter = 0
counter_df = 1
liste = []
for aktie in aktien:
    ticker = yf.Ticker(aktie)
    hist_data = ticker.history(period = "2y")
    print(hist_data)
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis = 1, inplace = True)

    # Holt-Winters-Modell initialisieren und anpassen
    model = ExponentialSmoothing(hist_data, trend = "add",
                                  seasonal = "add", initialization_method = "legacy-heuristic",seasonal_periods = 12)
    model_fit = model.fit()

    # Prognose für den Kurs des morgigen Tages erstellen
    forecast = model_fit.forecast(steps=10)
    print(forecast)
    forecast_list=forecast.tolist()
    day = 1
    #df = pd.DataFrame({"Gruppe(Nachname)": "Chevalaz-Wagener", "Aktie/Indize(Kürzel)": aktie, "Handelstag": 1, "Kurs": a}, index = [0])
    for i in forecast_list:
        a = round(i,2)
        b = ["Chevalaz-Wagener", aktie, day, a]
        liste.append(b)
        day = day+1
print(liste)
df = pd.DataFrame(liste, columns = ["Gruppe(Nachname)","Aktie/Indize(Kürzel)","Handelstag", "Kurs"])
#df.to_csv("prediction.csv", sep = ";", index = False)


            
