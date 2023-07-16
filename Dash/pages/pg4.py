import dash
from dash import dcc, html, callback, Output, Input, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import pandas as pd
import json
import numpy as np
from pandas_market_calendars import get_calendar
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from methods.Arima_func import Arima
pd.options.mode.chained_assignment = None

dash.register_page(__name__, name = "Arima prediciton")

card_plot_pred_arima = dbc.Card(
    [
        dbc.CardHeader(html.H5("Prediction Plot")),
        dbc.CardBody(
            [
                dcc.Graph(id = "plot_pred_arima", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_table_pred_arima = dbc.Card(
    [
        dbc.CardHeader(html.H5("Prediction table")),
        dbc.CardBody(
            [
                dash_table.DataTable(id = "data_table_pred_arima", page_size = 12, style_table = {"overflowX": "auto"}),
                html.Div(id = "output_arima")
            ]
        )
    ], className = "card border-primary mb-3"
)

card_plot_trainset_arima = dbc.Card(
    [
        dbc.CardHeader(html.H5("Comparison of predicted and true data")),
        dbc.CardBody(
            [
                dcc.Graph(id = "plot_trainset_arima", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_perf_arima = dbc.Card(
    [
        dbc.CardHeader(html.H5("Performance of trainset")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_perf_arima")
            ]
        )
    ], className = "card border-primary mb-3"
)

card_plot_pred_man_arima = dbc.Card(
    [
        dbc.CardHeader(html.H5("Manual prediction")),
        dbc.CardBody(
            [
                html.P("Select start day for prediction:"),
                dcc.DatePickerSingle(id = "datepicker_single_arima", min_date_allowed = date(1980,1,1),
                                    max_date_allowed = date.today() - timedelta(days=17), initial_visible_month = date.today() - timedelta(days=17), 
                                    date = date.today() - timedelta(days=17)),
                html.P(),
                dcc.Graph(id = "plot_pred_man_arima", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_perf_pred_man_arima = dbc.Card(
    [
        dbc.CardHeader(html.H5("Performance manual prediciton")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_perf_pred_man_arima")
            ]
        )
    ], className = "card border-primary mb-3"
)

layout = dbc.Container(
    [
        dbc.Row([
                html.P("Choose number of days to predict:"),
                dcc.Dropdown(id = "drop_days_arima", options = [1, 3, 5, 10, 30], value = 10, style = {"color": "black"}),
                html.P()
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_pred_arima
                ], width = 9
            ),
            dbc.Col(
                [
                    card_table_pred_arima
                ], width = 3
            )
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_trainset_arima
                ], width = 9
            ),
            dbc.Col(
                [
                    card_perf_arima
                ], width = 3
            )
        ]),
        html.Hr(style = {"margin-top": "-4px"}),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_pred_man_arima
                ], width = 9
            ),
            dbc.Col(
                [
                    card_perf_pred_man_arima
                ], width = 3
            )
        ]),
        dcc.Store(id = "forecast_store_arima"),
        dcc.Store(id = "forecast_fit_store_arima")
    ], fluid = True
)

@callback(
    Output("forecast_store_arima", "data"),
    Output("forecast_fit_store_arima", "data"),
    Input("data_store", "data"))
def update_Store_Data_arima(data):
    hist_data = pd.read_json(data, orient = "split")
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    today = datetime.today().date()
    start = today - timedelta(days = 730)
    hist_data = hist_data.loc[hist_data.index >= np.datetime64(start)]
    forecast_df, fit_values = Arima(hist_data, 30)
    forecast = forecast_df["Close"].tolist()
    market_calendar = get_calendar("NYSE")
    last_date = hist_data.index[-1]
    next_dates = market_calendar.valid_days(start_date=last_date + timedelta(days = 1), end_date = last_date + timedelta(days = 60))
    next_days_needed = next_dates[:30]
    for i in range(30):
        next_day = next_days_needed[i]
        next_day_date = next_day.date()
        hist_data.loc[np.datetime64(next_day_date)] = forecast[i]
    fitted_df = pd.DataFrame({"Close": fit_values}, index=fit_values.index)
    return hist_data.to_json(orient = "split"), fitted_df.to_json(orient = "split")

@callback(
    Output("data_table_pred_arima", "data"),
    Output("data_table_pred_arima", "columns"),
    Output("data_table_pred_arima", "style_data_conditional"),
    Output("data_table_pred_arima", "style_header"),
    Input("forecast_store_arima", "data"),
    Input("data_store", "data"),
    Input("drop_days_arima", "value"))
def update_Pred_Table_arima(forecast_data, data, count_days):
    hist_data = pd.read_json(data, orient = "split")
    fore_data = pd.read_json(forecast_data, orient = "split")
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    last_element = hist_data.index[-1]
    df_pred = fore_data.loc[fore_data.index > last_element]
    data_for_table = df_pred.head(count_days)
    data_for_table["Date"] = data_for_table.index
    data_for_table_round = data_for_table.round(2)
    data_for_table_round["Date"] = data_for_table_round["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    data_for_table_round = data_for_table_round[data_for_table_round.columns[-1:].tolist() + data_for_table_round.columns[:-1].tolist()]
    data_for_table_round.rename(columns = {"Close": "Predicted Price"}, inplace = True)
    columns = [{"name": col, "id": col} for col in data_for_table_round.columns]
    data_table = data_for_table_round.to_dict("records")
    style_data_conditional = [
        {
            "if": {"column_id": col},
            "color": "black",
        }
        for col in data_for_table_round.columns
    ]
    style_header = {"color": "black"}
    return data_table, columns, style_data_conditional, style_header

@callback(
    Output("plot_pred_arima", "figure"),
    Input("forecast_store_arima", "data"),
    Input("data_store", "data"),
    Input("drop_days_arima", "value"),
    Input("ticker_store", "data"))
def update_PlotPred_arima(forecast_data, data, count_days, ticker):
    hist_data = pd.read_json(data, orient = "split")
    fore_data = pd.read_json(forecast_data, orient = "split")
    ticker_data = json.loads(ticker)
    ticker_data_dic = dict(ticker_data)
    currency = ticker_data_dic["currency"]
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    hist_data = hist_data.iloc[-60:]
    last_element = hist_data.index[-1]
    df_pred = fore_data.loc[fore_data.index > last_element]
    data_pred = df_pred.head(count_days)
    merged_df = pd.concat([hist_data, data_pred])
    df_pred_add_last_element = merged_df.tail(count_days+1)
    fig_pred_arima = px.line(template = "simple_white")
    fig_pred_arima.add_trace(go.Scatter(x = hist_data.index, y = hist_data["Close"], mode = "lines", name = "Data", line_color = "blue"))
    fig_pred_arima.add_trace(go.Scatter(x = df_pred_add_last_element.index, y = df_pred_add_last_element["Close"],mode = "lines", name = "Prediction", line_color = "red"))
    fig_pred_arima.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}")
    return fig_pred_arima

@callback(
    Output("plot_trainset_arima", "figure"),
    Output("output_div_perf_arima", "children"),
    Input("forecast_fit_store_arima", "data"),
    Input("data_store", "data"),
    Input("ticker_store", "data"))
def update_TrainPlot_Perf_arima(fit_data, data, ticker):
    hist_data = pd.read_json(data, orient = "split")
    fitted_data = pd.read_json(fit_data, orient = "split")
    ticker_data = json.loads(ticker)
    ticker_data_dic = dict(ticker_data)
    currency = ticker_data_dic["currency"]
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    today = date.today()
    start = today - timedelta(days = 730)
    hist_data = hist_data.loc[start:]
    hist_data = hist_data.iloc[1:]
    fitted_data = fitted_data.iloc[1:]
    mae = round(mean_absolute_error(hist_data["Close"], fitted_data["Close"]), 2)
    mae_scaled = round((mae/hist_data["Close"].mean())*100, 2)
    mse = round(mean_squared_error(hist_data["Close"], fitted_data["Close"]), 2)
    rmse = round(math.sqrt(mse), 2)
    output = [
        html.P(f"Mean Absolute Error: {mae}"),
        html.P(f"Mean Absolute Error Scaled: {mae_scaled} %"),
        html.P(f"Mean Squared Error: {mse}"),
        html.P(f"Rooted Mean Squared Error: {rmse}")
    ]
    fig_train_arima = px.line(template = "simple_white")
    fig_train_arima.add_trace(go.Scatter(x = hist_data.index, y = hist_data["Close"], mode = "lines", name = "True Data", line_color = "blue"))
    fig_train_arima.add_trace(go.Scatter(x = fitted_data.index, y = fitted_data["Close"], mode = "lines", name = "Fitted Data", line_color = "red"))
    fig_train_arima.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}")
    return fig_train_arima, output

@callback(
    Output("plot_pred_man_arima", "figure"),
    Output("output_div_perf_pred_man_arima", "children"),
    Input("data_store", "data"),
    Input("ticker_store", "data"),
    Input("datepicker_single_arima", "date"))
def update_Pred_Man_arima(data, ticker, date):
    hist_data = pd.read_json(data, orient = "split")
    hist_data_for_pred = pd.read_json(data, orient = "split")
    ticker_data = json.loads(ticker)
    ticker_data_dic = dict(ticker_data)
    currency = ticker_data_dic["currency"]
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    hist_data_for_pred.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    date_format = datetime.strptime(date, "%Y-%m-%d").date()
    start = date_format - timedelta(days = 730)
    hist_data = hist_data.loc[hist_data.index >= np.datetime64(start)]
    hist_data_for_pred = hist_data_for_pred.loc[hist_data_for_pred.index >= np.datetime64(start)]
    predicted_day = datetime.strptime(date, "%Y-%m-%d").date()
    hist_data = hist_data.loc[hist_data.index < np.datetime64(predicted_day)]
    hist_data_for_pred = hist_data_for_pred.loc[hist_data_for_pred.index >= np.datetime64(predicted_day)]
    hist_data_for_pred = hist_data_for_pred.iloc[:10]
    forecast_df, fit_values = Arima(hist_data, 10)
    forecast = forecast_df["Close"].tolist()
    market_calendar = get_calendar("NYSE")
    last_date = hist_data.index[-1]
    next_dates = market_calendar.valid_days(start_date = last_date + timedelta(days = 1), end_date = last_date + timedelta(days = 20))
    next_days_needed = next_dates[:10]
    for i in range(10):
        next_day = next_days_needed[i]
        next_day_date = next_day.date()
        hist_data.loc[np.datetime64(next_day_date)] = forecast[i]
    hist_data = hist_data.iloc[-10:]
    fig_man_pred_arima = px.line(template = "simple_white")
    fig_man_pred_arima.add_trace(go.Scatter(x = hist_data.index.strftime("%Y-%m-%d"), y = hist_data["Close"], mode = "lines", name = "Prediction", line_color = "red"))
    fig_man_pred_arima.add_trace(go.Scatter(x = hist_data_for_pred.index.strftime("%Y-%m-%d"), y = hist_data_for_pred["Close"], mode = "lines", name = "True Data", line_color = "blue"))
    fig_man_pred_arima.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}", title = "Prediction", xaxis= {"type": "category"})
    
    mae = round(mean_absolute_error(hist_data_for_pred["Close"], hist_data["Close"]), 2)
    mae_scaled = round((mae/hist_data_for_pred["Close"].mean())*100, 2)
    mse = round(mean_squared_error(hist_data_for_pred["Close"], hist_data["Close"]), 2)
    rmse = round(math.sqrt(mse), 2)
    output = [
        html.P(f"Mean Absolute Error: {mae}"),
        html.P(f"Mean Absolute Error Scaled: {mae_scaled} %"),
        html.P(f"Mean Squared Error: {mse}"),
        html.P(f"Rooted Mean Squared Error: {rmse}")
    ]
    return fig_man_pred_arima, output
