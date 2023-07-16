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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from methods.LSTM_OS_func import evaluate_model, LSTM_OS, LSTM_OS_man_pred
pd.options.mode.chained_assignment = None

dash.register_page(__name__, name = "LSTM-OneShot prediction")

"""
Layout nahezu identisch zu pg3 (ein Plot weniger).
Callbacks nahezu identisch zu pg5 (Unterschiede wurden kommentiert)
"""

card_plot_pred_lstmos = dbc.Card(
    [
        dbc.CardHeader(html.H5("Prediction plot")),
        dbc.CardBody(
            [         
                dcc.Graph(id = "plot_pred_lstmos", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_table_pred_lstmos = dbc.Card(
    [
        dbc.CardHeader(html.H5("Prediction table")),
        dbc.CardBody(
            [
                dash_table.DataTable(id = "data_table_pred_lstmos", page_size = 12, style_table = {"overflowX": "auto"}),
                html.Div(id = "output_lstmos")
            ]
        )
    ], className = "card border-primary mb-3"
)

card_perf_lstmos = dbc.Card(
    [
        dbc.CardHeader(html.H5("Performance of train/validation set")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_perf_lstmos")
            ]
        )
    ], className = "card border-primary mb-3"
)

card_plot_pred_man_lstmos = dbc.Card(
    [
        dbc.CardHeader(html.H5("Manual prediction")),
        dbc.CardBody(
            [
                html.P("Select start day for prediction:"),
                dcc.DatePickerSingle(id = "datepicker_single_lstmos", min_date_allowed = date(1980, 1, 1), 
                                    max_date_allowed = date.today() - timedelta(days = 17), initial_visible_month = date.today() - timedelta(days = 17),
                                    date = date.today() - timedelta(days = 17)),
                html.P(),
                dcc.Graph(id = "plot_pred_man_lstmos", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_perf_pred_man_lstmos = dbc.Card(
    [
        dbc.CardHeader(html.H5("Performance manual prediciton")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_perf_pred_man_lstmos")
            ]
        )
    ], className = "card border-primary mb-3"
)

layout = dbc.Container(
    [
        dbc.Row([
            html.P("Choose number of days to predict:"),
            dcc.Dropdown(id = "drop_days_lstmos", options = [1, 3, 5, 10, 30], value = 10, style = {"color": "black"}),
            html.P()
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_pred_lstmos
                ], width = 9
            ),
            dbc.Col(
                [
                    card_table_pred_lstmos
                ], width = 3
            )
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_perf_lstmos
                ]
            )
        ]),
        html.Hr(style = {"margin-top": "-4px"}),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_pred_man_lstmos
                ], width = 9
            ),
            dbc.Col(
                [
                    card_perf_pred_man_lstmos
                ], width = 3
            )
        ]),
        dcc.Store(id = "forecast_store_lstmos"),
    ], fluid = True
)

@callback( #Führe anhand LSTM-OS Prognose aus, speichere diese im Store ab und update Performance Div
    Output("output_div_perf_lstmos", "children"),
    Output("forecast_store_lstmos", "data"),
    Input("data_store", "data"),
    Input("drop_days_lstmos", "value"),
    Input("token", "value"))
def update_TrainValPerf_StorePred_lstmos(data, count_days, token):
    hist_data = pd.read_json(data, orient = "split")
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    today = datetime.today().date()
    start = today - timedelta(days = 730)
    hist_data = hist_data.loc[hist_data.index >= np.datetime64(start)]
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_scaled = scaler.fit_transform(hist_data) #Skaliere Daten
    X, y = [], []
    window_size = 30
    prediction_size = count_days
    for i in range(window_size, len(data_scaled) - prediction_size + 1): #Erstelle Windowed-DF
        X.append(data_scaled[i-window_size: i])
        y.append(data_scaled[i: i+prediction_size])
    X, y = np.array(X), np.array(y)
    train_split = int(len(X) * 0.8) #Splitte Daten in Training und Validierung
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:] 
    y_val = y[train_split:]
    last_prices = data_scaled[-1 * window_size:]
    
    train_predictions, val_predictions, forecast_df = LSTM_OS(X_train, y_train, X_val, y_val, window_size,
                                                                prediction_size, last_prices, scaler, token, count_days)

    train_predictions = scaler.inverse_transform(train_predictions)
    val_predictions = scaler.inverse_transform(val_predictions)
    y_train_shaped = y_train.reshape(len(y_train), prediction_size)
    y_val_shaped = y_val.reshape(len(y_val), prediction_size)
    y_train_scaled = scaler.inverse_transform(y_train_shaped)
    y_val_scaled = scaler.inverse_transform(y_val_shaped)
    total_rmse_train, average_price_train, total_mae_train = evaluate_model(y_train_scaled, train_predictions) #Errechne Kennzahlen
    total_rmse_val, average_price_val, total_mae_val = evaluate_model(y_val_scaled, val_predictions)
    #Berrechne weitere Kennzahlen
    mae_train = round(total_mae_train, 2)
    mae_scaled_train = round((total_mae_train/average_price_train)*100, 2)
    mse_train = round((total_rmse_train**2), 2)
    mae_valid = round(total_mae_val, 2)
    mae_scaled_valid = round((total_mae_val/average_price_val)*100, 2)
    mse_valid = round((total_rmse_val**2), 2)
    output = [
        html.P(f"MAE: {mae_train:.2f} (Train) {mae_valid:.2f} (Validation)"),
        html.P(f"MAE Scaled: {mae_scaled_train} % (Train) {mae_scaled_valid} % (Validation)"),
        html.P(f"MSE: {mse_train:.2f} (Train) {mse_valid:.2f} (Validation)"),
        html.P(f"RMSE: {total_rmse_train:.2f} (Train) {total_rmse_val:.2f} (Validation)")
    ]
    forecast = forecast_df["Close"].tolist()
    last_date = hist_data.index[-1]
    market_calendar = get_calendar("NYSE")
    next_dates = market_calendar.valid_days(start_date = last_date + timedelta(days = 1), end_date = last_date + timedelta(days = 60))
    next_days_needed = next_dates[:30]
    for i in range(count_days):
        next_day = next_days_needed[i]
        next_day_date = next_day.date()
        hist_data.loc[np.datetime64(next_day_date)] = forecast[i]
    hist_data = hist_data.iloc[-1* count_days:]
    return output, hist_data.to_json(orient = "split")

@callback(Output("data_table_pred_lstmos", "data"),
    Output("data_table_pred_lstmos", "columns"),
    Output("data_table_pred_lstmos", "style_data_conditional"),
    Output("data_table_pred_lstmos", "style_header"),
    Input("forecast_store_lstmos", "data"),
    Input("data_store", "data"))
def update_PredTable_lstmos(forecast_data, data):
    hist_data = pd.read_json(data, orient = "split")
    fore_data = pd.read_json(forecast_data, orient = "split")
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    last_element = hist_data.index[-1]
    data_for_table = fore_data.loc[fore_data.index > last_element]
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
    Output("plot_pred_lstmos", "figure"),
    Input("forecast_store_lstmos", "data"),
    Input("data_store", "data"),
    Input("ticker_store", "data"))
def update_Plot_Pred_lstmos(forecast_data, data, ticker):
    hist_data = pd.read_json(data, orient = "split")
    fore_data = pd.read_json(forecast_data, orient = "split")
    ticker_data = json.loads(ticker)
    ticker_data_dic = dict(ticker_data)
    currency = ticker_data_dic["currency"]
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    hist_data = hist_data.iloc[-60:]
    last_element = hist_data.index[-1]
    data_pred = fore_data.loc[fore_data.index > last_element]
    merged_df = pd.concat([hist_data, data_pred])
    df_pred_add_last_element = merged_df.tail(len(data_pred)+1)
    #Erstelle Plot
    fig_pred_lstmos = px.line(template = "simple_white")
    fig_pred_lstmos.add_trace(go.Scatter(x = hist_data.index, y = hist_data["Close"], mode = "lines", name = "Data", line_color = "blue"))
    fig_pred_lstmos.add_trace(go.Scatter(x = df_pred_add_last_element.index, y = df_pred_add_last_element["Close"],mode = "lines", name = "Prediction", line_color = "red"))
    fig_pred_lstmos.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}")
    return fig_pred_lstmos

@callback(
    Output("plot_pred_man_lstmos", "figure"),
    Output("output_div_perf_pred_man_lstmos", "children"),
    Input("data_store", "data"),
    Input("ticker_store", "data"),
    Input("datepicker_single_lstmos", "date"),
    Input("token", "value"))
def update_Pred_Man_lstmos(data, ticker, date, token):
    hist_data = pd.read_json(data, orient = "split")
    date_format = datetime.strptime(date, "%Y-%m-%d").date()
    start = date_format - timedelta(days=730)
    hist_data = hist_data.loc[hist_data.index >= np.datetime64(start)]
    ticker_data = json.loads(ticker)
    ticker_data_dic = dict(ticker_data)
    currency = ticker_data_dic["currency"]
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    hist_data_for_pred = hist_data.copy()
    predicted_day = datetime.strptime(date, "%Y-%m-%d").date()
    hist_data = hist_data.loc[hist_data.index < np.datetime64(predicted_day)]
    last_date = hist_data.index[-1]
    hist_data_for_pred = hist_data_for_pred.loc[hist_data_for_pred.index >= np.datetime64(predicted_day)]
    hist_data_for_pred = hist_data_for_pred.iloc[:10]
    data_for_func = hist_data.copy()

    forecast_df = LSTM_OS_man_pred(data_for_func, token, date)

    forecast = forecast_df["Close"].tolist()
    market_calendar = get_calendar("NYSE")
    next_dates = market_calendar.valid_days(start_date=last_date + timedelta(days = 1), end_date = last_date + timedelta(days = 20))
    next_days_needed = next_dates[:10]
    for i in range(10):
        next_day = next_days_needed[i]
        next_day_date = next_day.date()
        hist_data.loc[np.datetime64(next_day_date)] = forecast[i]
    hist_data = hist_data.iloc[-10:]
    #Erstelle Plot
    fig_man_pred_lstmos = px.line(template = "simple_white")
    fig_man_pred_lstmos.add_trace(go.Scatter(x = hist_data.index.strftime("%Y-%m-%d"), y = hist_data["Close"], mode = "lines", name = "Prediction", line_color = "red"))
    fig_man_pred_lstmos.add_trace(go.Scatter(x = hist_data_for_pred.index.strftime("%Y-%m-%d"), y = hist_data_for_pred["Close"], mode = "lines", name = "True Data", line_color = "blue"))
    fig_man_pred_lstmos.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}", title = "Prediction", xaxis= {"type": "category"})
    #Berechne Kennzahlen
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
    return fig_man_pred_lstmos, output
