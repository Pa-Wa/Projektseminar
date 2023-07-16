import dash
from dash import dcc, html, callback, Output, Input
from dash import html, dcc
import dash_bootstrap_components as dbc
import yfinance as yf
import json

'''
Achtung: Bei wiederholter Ausführung des Dashboards, bitte die Seite im Browser vorher schließen und nicht nur Refreshen!
Refresh löscht nicht den Cache, was bei den Inputs (bspw. Dropdowns) zu einem Error führen kann.

Main-Page: Legt Layout der einzelnen Seiten fest
'''

app = dash.Dash(__name__, use_pages = True, external_stylesheets = [dbc.themes.DARKLY]) #Bootstrap Stylesheet

#Erstelle Leiste in der die verschiedenen Seiten ausgewählt werden können
topbar = dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Overview", href = "/")),
                dbc.NavItem(dbc.NavLink("Analysis", href = "/pg2")),
                dbc.NavItem(dbc.NavLink("Holt-Winters prediciton", href = "/pg3")),
                dbc.NavItem(dbc.NavLink("ARIMA prediciton", href = "/pg4")),
                dbc.NavItem(dbc.NavLink("LSTM prediciton", href = "/pg5")),
                dbc.NavItem(dbc.NavLink("LSTM-One-Shot prediciton", href = "/pg6")),
                dbc.Col(html.P(""), width = 1), #Leeres Element, damit nächstes Element weiter am Rand steht
                dbc.Col(html.P("Enter US stock token:"), width = 1), #width passt die Breite des Elementes an
                dbc.Col(
                        [
                                dbc.Input(id = "token", placeholder = "Token", type = "text", value = "TSLA", debounce = True) #Manuelles Input-Feld zur Nutzereingabe des Aktien-Tokens
                        ], width= True
                )                                 
            ], vertical = False ,pills = True, className = "navbar navbar-expand-lg bg-primary" #className legt den Style der einzelnen Komponenten fest, bspw. Bar (hier) oder Cards (kommen noch)
)

#Erstelle das allgemeine Layout für jede Seite
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Finance-Board", className = "text-center text-primary", style = {"padding-top": "20px"})) #Überschrift
    ]),
    html.Hr(), #Horizontale Trennlinie
    dbc.Row(        
            dbc.Col(
                [
                    topbar #Fügt oben definierte Bar ein
                ])),
    html.Hr(),
    dbc.Row(
            dbc.Col(
                [
                    dash.page_container #Fügt definierte Seitennamen und Referenzen ein
                ])),
    dbc.Row(
        dbc.Col(
             [
                 dcc.Store(id = "data_store"), #Speicher für geladene historische Kursdaten, ist somit auf jeder Seite direkt verfügbar
                 dcc.Store(id = "ticker_store") #Speicher des geladenen Aktien Tickers (Informationen über die gewählte Aktie)
             ]
        ))
], fluid = True) #Elemente im Dashboard passen sich automatisch an die verfügbare Bildschirmgröße an 

@callback( #Lädt hist. Daten und Ticker der gewählten Aktie und speichere diese im Store ab
    Output("data_store", "data"),
    Output("ticker_store", "data"),
    Input("token", "value"))
def store_data(value):
    aktie_ticker = yf.Ticker(value)
    df = yf.download(value, period = "max")
    #ticker_data_json = json.dumps(aktie_ticker.info)
    fast_info = aktie_ticker.fast_info
    fast_info_dict = fast_info.items()
    fast_info_json = json.dumps(fast_info_dict)
    return df.to_json(orient = "split"), fast_info_json #Im Speicher können nur JSON-Dateien gespeichert werden, deshalb Formatierung

if __name__ == "__main__": #Führt Dashboard aus
    app.run(debug=False)