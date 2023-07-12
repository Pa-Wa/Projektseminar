'''
Fassung, die Anfang des Semester präsentiert wurde
'''

from dash import Dash, html, dcc, Input, Output
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from datetime import date, timedelta
import dash_mantine_components as dmc
import dash_daq as daq

app = Dash(__name__)

#df = pd.DataFrame()

dictonary_period = {0: "1d", 1: "3d", 2: "5d", 3: "1mo", 4: "3mo", 5: "6mo", 6: "1y", 
                    7: "2y", 8: "5y", 9: "10y", 10:"max"}

app.layout = html.Div(children = [
    html.H1(children = "Finance Dashboard", style={"textAlign": "center"}),
    html.H2(children = "First Try", style = {"textAlign": "center"}),
    html.Div(children = [
    html.H2(children = "Filter")]),
    html.Div(children = [
    html.P(children = "Enter Token: ", style = {"width": "800", "display": "inline"}),
    dcc.Input(id = "token", placeholder = "Token", type = "text", value = "TSLA", debounce=True, style = {"width": "800", "display": "inline", "margin-right": 15}),
    html.P(children = "Choose Type: ", style = {"width": "800", "display": "inline"}),
    dcc.RadioItems(id = "RI", options = ["Close", "Volume", "Low" ], value = "Close", style= {"width": "800", "display": "inline"}, inline = True)],style = {"width": "800", "display": "inline"}),
    html.P(),
    html.P(children = "Choose Period:"),
    html.Div(children = [
    daq.Slider(min = 0, max = 10, step = None,
               marks = {0: "1d", 1: "3d", 2: "5d", 3: "1mo", 4: "3mo", 5: "6mo", 6: "1y", 
                7: "2y", 8: "5y", 9: "10y", 10:"max"}, value = 10, id = "selected_period")]),
    html.H1(),
    html.Div(children = [
    dmc.Checkbox(id = "checkbox_range", label = "manual date input", mb=10, style= {"width": "200", "display": "inline"}),
    html.P(),
    dcc.DatePickerRange(id = "selected_date_range", min_date_allowed = date(1980,1,1), max_date_allowed = date.today(),
              initial_visible_month = date.today() - timedelta(days=7), start_date = date.today() - timedelta(days=7), end_date = date.today(), style= {"width": "800", "display": "inline"})]),
    html.Hr(style= {"border-width": "10px;"}),
    html.Div(children = [
    html.H2(children = "Overview"),
    html.Div(children = [
    html.P(id = "Name1", children = "Name: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "name", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "stadt1", children = "City: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "stadt", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "land1", children = "Country: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "land", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "webseite1", children = "Website: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "webseite", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "industry1", children = "Industy: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "industry", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "sektor1", children = "Sector: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "sektor", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "anz_mitarbeiter1", children = "Employees: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "anz_mitarbeiter", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "currency1", children = "Currency: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "currency", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "kurs1", children = "Price: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "kurs", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "kurs_vortag1", children = "Price day before: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "kurs_vortag", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "volumen1", children = "Volume: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "volumen", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "marktkapital1", children = "Market Capital: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "marktkapital", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "unternehmenswert1", children = "Enterprise Value: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "unternehmenswert", style = {"width": "800", "display": "inline"})]),
    html.Div(children = [
    html.P(id = "gl_average1", children = "sl. average 50d: ", style = {"width": "800", "display": "inline"}),
    html.P(id = "gl_average", style = {"width": "800", "display": "inline"})])]),
    html.Hr(style= {"border-width": "10px;"}),
    html.H2(children = "Visualize"),
    dcc.Graph(figure = {}, id = "kurs_plot"),
    html.Hr(style= {"border-width": "10px;"}),
    html.H2(children = "Analysis"),
    dcc.RadioItems(id = "min_max_average", options = ["Average", "Minimum", "Maximum"], value = "Average"),
    html.P(),
    html.Div(id = "MinMaxAverage"),
    html.Hr(style= {"border-width": "10px;"}),
    html.H2(children = "Forecast"),
    html.P(children = "Forecast for tomorrow (moving average):"),
    html.Div(id = "Forecast")
    ])

@app.callback(
    Output(component_id = "kurs_plot", component_property = "figure"),
    Input(component_id = "token", component_property = "value"),
    Input(component_id = "RI", component_property = "value"),
    Input(component_id = "selected_period", component_property= "value"),
    Input(component_id = "checkbox_range", component_property= "checked"),
    Input(component_id = "selected_date_range", component_property= "start_date"),
    Input(component_id = "selected_date_range", component_property= "end_date")
)

def update_graph_forecast (selected_token, selected_obj, selected_period, checked, start_date, end_date):
    ticker = yf.Ticker(selected_token)
    selected_period_dic = dictonary_period[selected_period]
    if checked:
        hist_data = ticker.history(start = start_date, end = end_date)
    else:
        if selected_period_dic == "1d" or selected_period_dic == "3d" or selected_period_dic == "5d":
            hist_data = ticker.history(period = selected_period_dic, interval = "60m")
        else:
            hist_data = ticker.history(period = selected_period_dic)
    
    company_name = ticker.info["longName"]
    #global df    
    #df = hist_data
    
    if selected_obj == "Close" or selected_obj == "Low":
        fig = px.line(x=hist_data.index, y = hist_data[selected_obj], labels= {"x": "Date", "y": f"{selected_obj} Price"}, title = f"{company_name}")
    if selected_obj == "Volume":
        fig = px.line(x=hist_data.index, y = hist_data[selected_obj], labels= {"x": "Date", "y": "Volume"}, title = f"{company_name}")
    """
    trace = go.Scatter(x = hist_data.index, y = hist_data[selected_obj])
    layout = go.Layout(title=f'{company_name} {selected_obj}')
    fig = go.Figure(data=[trace], layout=layout)
    """
    return fig
    

@app.callback(
    Output(component_id = "MinMaxAverage", component_property = "children"),
    Input(component_id = "token", component_property = "value"),
    Input(component_id = "RI", component_property = "value"),
    Input(component_id = "selected_period", component_property= "value"),
    Input(component_id = "checkbox_range", component_property= "checked"),
    Input(component_id = "selected_date_range", component_property= "start_date"),
    Input(component_id = "selected_date_range", component_property= "end_date"),
    Input(component_id = "min_max_average", component_property= "value")
    )

def update_MinMaxAverage (selected_token, selected_obj, selected_period, checked, start_date, end_date, option_min_max):
    ticker = yf.Ticker(selected_token)
    selected_period_dic = dictonary_period[selected_period]
    if checked:
        hist_data = ticker.history(start = start_date, end = end_date)
    else:
        if selected_period_dic == "1d" or selected_period_dic == "3d" or selected_period_dic == "5d":
            hist_data = ticker.history(period = selected_period_dic, interval = "60m")
        else:
            hist_data = ticker.history(period = selected_period_dic)
    if option_min_max == "Minimum":
        minmax = min(hist_data[selected_obj])
    if option_min_max == "Maximum":
        minmax = max(hist_data[selected_obj])
    if option_min_max == "Average":
        minmax = hist_data[selected_obj].mean()
    currency = ticker.fast_info["currency"]
    return f"The {option_min_max} Price in the choosen period is {minmax:.3f} {currency}."

@app.callback(
    Output(component_id = "Forecast", component_property = "children"),
    Input(component_id = "token", component_property = "value"),
    )

def update_forecast (selected_token):
    ticker = yf.Ticker(selected_token)
    hist_data = ticker.history(period = "max")
    currency = ticker.fast_info["currency"]
    fiftydaydata = hist_data.tail(50)
    forecast = fiftydaydata["Close"].mean()
    return f"{forecast:.3f} {currency}"

@app.callback(
    Output(component_id = "name", component_property = "children"),
    Output(component_id = "stadt", component_property = "children"),
    Output(component_id = "land", component_property = "children"),
    Output(component_id = "webseite", component_property = "children"),
    Output(component_id = "industry", component_property = "children"),
    Output(component_id = "sektor", component_property = "children"),
    Output(component_id = "anz_mitarbeiter", component_property = "children"),
    Output(component_id= "currency", component_property = "children"),
    Output(component_id= "kurs", component_property = "children"),
    Output(component_id= "kurs_vortag", component_property = "children"),
    Output(component_id= "volumen", component_property = "children"),
    Output(component_id= "marktkapital", component_property = "children"),
    Output(component_id= "unternehmenswert", component_property = "children"),
    Output(component_id= "gl_average", component_property = "children"),
    Input(component_id = "token", component_property = "value")
)

def update_übersicht(selected_token):
    ticker = yf.Ticker(selected_token)
    company_name = ticker.info["longName"]
    stadt = ticker.info["city"]
    land = ticker.info["country"]
    webseite = ticker.info["website"]
    industry = ticker.info["industry"]
    sektor = ticker.info["sector"]
    anz_mitarbeiter = ticker.info["fullTimeEmployees"]
    currency = ticker.fast_info["currency"]
    kurs = ticker.info["currentPrice"]
    kurs_vortag = ticker.fast_info["previousClose"]
    volume = ticker.fast_info["lastVolume"]
    marktkapital = ticker.info["marketCap"]
    unternehmenswert = ticker.info["enterpriseValue"]
    gl_average = ticker.fast_info["fiftyDayAverage"]
    return company_name, stadt, land, webseite, industry, sektor, anz_mitarbeiter, currency, kurs, kurs_vortag, volume, marktkapital, unternehmenswert, f"{gl_average:.2f}"


if __name__ == "__main__":
    app.run_server(debug = True)
