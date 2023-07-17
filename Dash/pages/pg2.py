import dash
from dash import dcc, html, callback, Output, Input, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import pandas as pd
import json
import numpy as np

dash.register_page(__name__, name = "Analysis")

#Definiere Listen und Dicts für spätere Dropdowns bzw. Datenübergabe
list_col = ["Open", "High", "Low", "Close", "Adj Close", "Volume"] 

dictonary_period = {0: "7d", 1: "2w", 2: "1mo", 3: "3mo", 4: "6mo", 5: "1y", 
                    6: "2y", 7: "5y", 8: "10y", 9:"max"}
dictonary_period_todays = {0: 7, 1: 14, 2: 30, 3: 90, 4: 180, 5: 365, 
                    6: 730, 7: 1825, 8: 3650, 9: "max"}

#Definiere Cards für die einzelnen Elemente
card_timehorizont_analysis = dbc.Card(
    [
        dbc.CardHeader(html.H5("Period settings")),
        dbc.CardBody(
            [
                html.P("Choose period:"), #normaler Text
                dcc.Slider(min = 0, max = 9, step = None, marks = {0: "7d", 1: "2w", 2: "1mo", 3: "3mo", 4: "6mo", 5: "1y", 6: "2y", 7: "5y", 8: "10y", 9: "max"}, value = 9, id = "selected_period_analysis"), #Slider, mit dem man verschiedene Perioden wählen kann
                html.P(), #Leerzeile
                dbc.Switch(id = "checkbox_range_analysis", label = "Check for manual date input"), #Checkbox, die man anklicken kann wenn man selbst eine Range eingeben möchte
                dcc.DatePickerRange(id = "selected_date_range_analysis", min_date_allowed = date(1980,1,1), max_date_allowed = date.today(),
                    initial_visible_month = date.today() - timedelta(days=7), start_date = date.today() - timedelta(days=7), end_date = date.today(), style= {"width": "800", "display": "inline"}), #Range-Picker für manuelle Datums-Eingabe
            ]
        )
    ], className = "card border-primary mb-3"
)

card_man_plot_analysis = dbc.Card(
    [
        dbc.CardHeader(html.H5("Development of stock prices")),
        dbc.CardBody(
            [
                html.P("Choose object:"),
                dbc.RadioItems(id = "RI_analysis", options = ["Open", "Close", "Low", "High", "Volume"], value = "Close", inline = True), #RadioItems, wo man eine der Möglichkeiten anklicken kann
                dcc.Graph(id = "plot_man_analysis", figure={}) #Plot
            ]
        )
    ], className = "card border-primary mb-3"
)

card_table_analysis = dbc.Card(
    [
        dbc.CardHeader(html.H5("Historical data")),
        dbc.CardBody(
            [
                dash_table.DataTable(id = "data_table_analysis", page_size = 40, style_table = {"overflowX": "auto"}), #Tabelle
                html.Div(id = "output_analysis") #"Output" wird für Tabelle benötigt
            ]
        )
    ], className = "card border-primary mb-3"
)

card_today_data_analysis = dbc.Card(
    [
        dbc.CardHeader(html.H5("Information today")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_today_analysis") #Div, indem die Informationen angezeigt werden
            ]
        )
    ], className = "card border-primary mb-3"
)

card_time_data_analysis = dbc.Card(
    [
        dbc.CardHeader(html.H5("Information period")),
        dbc.CardBody(
            [
                html.Div(id = "output_div_period_analysis") #Div, indem die Informationen angezeigt werden
            ]
        )
    ], className = "card border-primary mb-3"
)

# Definiere Layout der Seite
layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(
                [
                    card_timehorizont_analysis
                ]
            )
        ]),
        html.Hr(style = {"margin-top": "-4px"}),
        dbc.Row([
            dbc.Col(
                [
                    card_man_plot_analysis
                ], width = 9 #width passt die Breite des Elementes an, jede Row hat eine Breite von 12
            ),
            dbc.Col(
                [
                    dbc.Container(
                        [
                        dbc.Row(
                            [
                                card_today_data_analysis
                            ]
                        ),
                        html.P(),
                        dbc.Row(
                            [
                                card_time_data_analysis
                            ]
                        )
                        ]
                    )
                ], width = 3    
            )
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_table_analysis
                ]
            )
        ]),
        dcc.Store(id = "store_filtered_data_analysis") #Speichert gefilterte hist. Daten
    ], fluid = True
)

@callback( #Daten je nach manueller Eingabe (Slider/Range-Picker) filtern und in Store abspeichern
    Output("store_filtered_data_analysis", "data"),
    Input("data_store", "data"),
    Input("selected_period_analysis", "value"),
    Input("checkbox_range_analysis", "value"),
    Input("selected_date_range_analysis", "start_date"),
    Input("selected_date_range_analysis", "end_date"))
def update_Data_analysis(data, period, check, start_date, end_date):
    hist_data = pd.read_json(data, orient = "split")
    selected_period_str = dictonary_period[period]
    if check: #Falls Eingabe über Range-Picker
        start = datetime.strptime(start_date, "%Y-%m-%d").date() #Wandelt String-Date in Date-Format um
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        hist_data = hist_data.loc[hist_data.index >= np.datetime64(start)] #Filtert Datensatz nach Anfangs- und Enddatum
        hist_data = hist_data.loc[hist_data.index <= np.datetime64(end)]
    else:
        if selected_period_str != "max":
            today = date.today()
            start = today - timedelta(days = dictonary_period_todays[period]) #Passt Start an, indem vom heutigen Tag die gewähle Entfernung abgezogen wird
            hist_data = hist_data.loc[start:]
    return hist_data.to_json(orient = "split") 

@callback( #Aktualisiert die "time_delta"-Card (je nach manueller Eingabe)
        Output("output_div_period_analysis","children"),
        Input("store_filtered_data_analysis", "data"),
        Input("ticker_store", "data"),
        Input("RI_analysis", "value"))
def update_Period_analysis(data, ticker, selected_obj):
    ticker_data = json.loads(ticker)
    hist_data = pd.read_json(data, orient = "split")
    hist_data.drop(columns = ["Adj Close"], inplace = True) #Entferne nicht benötigte Spalten aus dem DataFrame
    currency = ticker_data["financialCurrency"]
    period_high = round(hist_data[selected_obj].max(), 2) #Berechnet Kennzahlen
    period_low = round(hist_data[selected_obj].min(), 2)
    period_mean = round(hist_data[selected_obj].mean(), 2)
    period_std = round(hist_data[selected_obj].std(), 2)
    if selected_obj in ["Open", "High", "Low", "Close"]: #Currency wird nicht bei Volumen benötigt
        output = [
            html.P(f"Average: {period_mean} {currency}"),
            html.P(f"Low: {period_low} {currency}"),
            html.P(f"High: {period_high} {currency}"),
            html.P(f"Standard Deviation: {period_std}")
        ]
    else:
        output = [
            html.P(f"Average: {period_mean}"),
            html.P(f"Low: {period_low}"),
            html.P(f"High: {period_high}"),
            html.P(f" Standard Deviation: {period_std}")
        ]
    return output

@callback( #Aktualisiert die "today_data"-Card
    Output("output_div_today_analysis", "children"),
    Input("ticker_store", "data"))
def update_Today_analysis(ticker):
    ticker_data = json.loads(ticker)
    open_price = ticker_data["regularMarketOpen"]
    close_price_yesterday = ticker_data["regularMarketPreviousClose"]
    low_price = ticker_data["regularMarketDayLow"]
    high_price = ticker_data["regularMarketDayHigh"]
    currency = ticker_data["financialCurrency"]
    current_price = ticker_data["currentPrice"]
    price_trend = round(((current_price - close_price_yesterday) / close_price_yesterday) * 100, 2) #Berechnet, ob Kurs gefallen oder gestiegen, passe demnach das Vorzeichen an
    if price_trend < 0:
        trend = "-" 
        price_trend = price_trend * (-1)
    else:
        trend = "+"
    output = [ #Output fürs Div innerhalb der Card
        html.P(f"Current Price: {current_price} {currency}"),
        html.P(f"Price Trend: {trend} {price_trend} %"),
        html.P(f"Previous Close: {close_price_yesterday} {currency}"),
        html.P(f"Open: {open_price} {currency}"),
        html.P(f"High: {low_price} {currency}"),
        html.P(f"Low: {high_price} {currency}"),
    ]
    return output

@callback( #Aktualisiert die Tabelle, in der die vergangenen Kurse angezeigt werden (je nach manueller Eingabe)
    Output("data_table_analysis", "data"),
    Output("data_table_analysis", "columns"),
    Output("data_table_analysis", "style_data_conditional"),
    Output("data_table_analysis", "style_header"),
    Input("store_filtered_data_analysis", "data"))
def update_Table_analysis(data):
    hist_data = pd.read_json(data, orient = "split")
    hist_data.drop(columns = ["Adj Close"], inplace = True)
    hist_data_round = hist_data.round(2) #Runde auf zwei Nachkommastellen
    hist_data_round["Date"] = hist_data_round.index
    hist_data_round["Date"] = hist_data_round["Date"].apply(lambda x: x.strftime("%Y-%m-%d")) #Ändert Date-Format zu einem String-Format
    hist_data_round = hist_data_round[hist_data_round.columns[-1:].tolist() + hist_data_round.columns[:-1].tolist()] #Ändert Reihenfolge der Spalten
    columns = [{"name": col, "id": col} for col in hist_data_round.columns]
    data_table = hist_data_round.to_dict("records") #Formatierung für Tabelle
    style_data_conditional = [ #Passe Style der Tabelle an
        {
            "if": {"column_id": col},
            "color": "black",
        }
        for col in hist_data_round.columns
    ]
    style_header = {"color": "black"}
    return data_table, columns, style_data_conditional, style_header          

@callback( #Aktualisiert den Plot, je nach manueller Eingabe 
    Output("plot_man_analysis", "figure"),
    Input("store_filtered_data_analysis", "data"),
    Input("RI_analysis", "value"),
    Input("ticker_store", "data"))
def update_Plot_analysis(data, selected_obj, ticker):
    hist_data = pd.read_json(data, orient = "split")
    ticker_data = json.loads(ticker)
    currency = ticker_data["financialCurrency"]
    company_name = ticker_data["longName"]
    if len (hist_data) >= 30: #Falls der gewählte Zeitraum mehr als 30 Kurse enthält, plotte mit Datumsangaben im Date-Format
        if selected_obj == "Close" or selected_obj == "Low" or selected_obj == "High" or selected_obj == "Open": #Falls nicht Volumen (für Beschriftung)
            fig_period = px.line(x=hist_data.index, y = hist_data[selected_obj], labels= {"x": "Date", "y": f"{selected_obj} Price in {currency}"}, template = "simple_white", title = f"{company_name}")
        else:
            fig_period = px.line(x=hist_data.index, y = hist_data[selected_obj], labels= {"x": "Date", "y": "Volume"}, template = "simple_white", title = f"{company_name}")
    else: #Falls weniger, dann plotte mit String-Datum und zeige nur die Tage auf der x-Achse an, die auch im DataFrame vorkommen -> keine Wochenendtage 
        if selected_obj == "Close" or selected_obj == "Low" or selected_obj == "High" or selected_obj == "Open":
            fig_period = px.line(x=hist_data.index.strftime("%Y-%m-%d"), y = hist_data[selected_obj], labels= {"x": "Date", "y": f"{selected_obj} Price in {currency}"}, template = "simple_white", markers=True, title = f"{company_name}")
            fig_period.update_layout(xaxis={"type": "category"})
        else:
            fig_period = px.line(x=hist_data.index, y = hist_data[selected_obj], labels= {"x": "Date", "y": "Volume"}, template = "simple_white", title = f"{company_name}")
    return fig_period
