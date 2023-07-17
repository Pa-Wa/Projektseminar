import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import json

dash.register_page(__name__, path = "/", name = "Overview") #Startseite

#Definiere Cards für die einzelnen Elemente
card_overview = dbc.Card(
    [
        dbc.CardHeader(html.H5("Stock information")), #Card Überschrift
        dbc.CardBody( #Card Inhalt
            [
                html.Div(children = [
                html.P(id = "name_fix", children = "Name: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "name", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "city_fix", children = "City: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "city", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "country_fix", children = "Country: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "country", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "webseite_fix", children = "Website: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "webseite", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "industry_fix", children = "Industy: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "industry", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "sector_fix", children = "Sector: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "sector", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "employees_fix", children = "Employees: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "employees", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "currency_fix", children = "Currency: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "currency", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "price_fix", children = "Price: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "price", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "price_day_before_fix", children = "Price day before: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "price_day_before", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "volume_fix", children = "Volume: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "volume", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "market_capital_fix", children = "Market capital: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "market_capital", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "enterprise_value_fix", children = "Enterprise value: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "enterprise_value", style = {"width": "800", "display": "inline"})]),
                html.Div(children = [
                html.P(id = "moving_average_fix", children = "Sliding average 50d: ", style = {"width": "800", "display": "inline"}),
                html.P(id = "moving_average", style = {"width": "800", "display": "inline"})]),
            ]
        )
    ], className = "card border-primary mb-3"
)

card_plot_overview = dbc.Card(
    [
        dbc.CardHeader(html.H5("Stock history")),
        dcc.Graph(id = "plot_overview", figure = {})
    ], className = "card border-primary mb-3"
)

#Definiere Layout der Seite
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(card_overview, width = 3), #Fügt Cards ein
                dbc.Col(card_plot_overview, width = 9)
            ],
        )
    ], fluid = True, 
)

@callback( #Lädt abgespeicherten Ticker und daraus die benötigten Informationen der Aktie und gibt sie an die oben definierte Card weiter
    Output("name", "children"),
    Output("city", "children"),
    Output("country", "children"),
    Output("webseite", "children"),
    Output("industry", "children"),
    Output("sector", "children"),
    Output("employees", "children"),
    Output("currency", "children"),
    Output("price", "children"),
    Output("price_day_before", "children"),
    Output("volume", "children"),
    Output("market_capital", "children"),
    Output("enterprise_value", "children"),
    Output("moving_average", "children"),
    Input("ticker_store", "data"))
def update_Overview_Div(ticker):
    ticker_data = json.loads(ticker)
    company_name = ticker_data["longName"] #Lädt aus Ticker die jeweilige Information der Aktie
    city = ticker_data["city"]
    country = ticker_data["country"]
    webseite = ticker_data["website"]
    industry = ticker_data["industry"]
    sector = ticker_data["sector"]
    employees = ticker_data["fullTimeEmployees"]
    currency = ticker_data["financialCurrency"]
    price = ticker_data["currentPrice"]
    price_before = ticker_data["regularMarketPreviousClose"]
    volume = ticker_data["volume"]
    market_capital = ticker_data["marketCap"]
    enterprise_value = ticker_data["enterpriseValue"]
    moving_average = ticker_data["fiftyDayAverage"]
    return company_name, city, country, webseite, industry, sector, employees, currency, price, price_before, volume, market_capital, enterprise_value, f"{moving_average:.2f}" 

@callback( #Lädt hist. Daten und plottet den gesamten Kursverlauf
    Output("plot_overview", "figure"),
    Input("data_store", "data"),
    Input("ticker_store", "data"))
def update_Overview_Plot(data, ticker):
    hist_data = pd.read_json(data, orient = "split")
    ticker_data = json.loads(ticker)
    company_name = ticker_data["longName"]
    currency = ticker_data["financialCurrency"]
    fig_max_period = px.line(x = hist_data.index, y = hist_data["Close"], labels = {"x": "Date", "y": f"Close Price in {currency}"}, template = "simple_white", title = f"{company_name}) #Linien-Plot
    return fig_max_period
