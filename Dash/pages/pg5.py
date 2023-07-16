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
from methods.LSTM_func import data_to_windowed_data, windowed_df_to_d_x_y, LSTM, LSTM_man_pred
pd.options.mode.chained_assignment = None

dash.register_page(__name__, name = "LSTM prediction")

"""
Layout identisch zu Holt_Winters (für Kommentierung s. pg3)
Callbacks sind kommentiert.
"""

card_plot_pred_lstm = dbc.Card(
    [
        dbc.CardHeader(html.H5("Prediction plot")),
        dbc.CardBody(
            [
                dcc.Graph(id = "plot_pred_lstm", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_table_pred_lstm = dbc.Card(
    [
        dbc.CardHeader(html.H5("Prediction table")),
        dbc.CardBody(
            [
                dash_table.DataTable(id = "data_table_pred_lstm", page_size = 12, style_table = {"overflowX": "auto"}),
                html.Div(id = "output_lstm")
            ]
        )
    ], className = "card border-primary mb-3"
)

card_plot_trainvalsets_lstm = dbc.Card(
    [
        dbc.CardHeader(html.H5("Comparison of predicted and true data")),
        dbc.CardBody(
            [
                dcc.Graph(id = "plot_trainvalsets_lstm", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_perf_lstm = dbc.Card(
    [
        dbc.CardHeader(html.H5("Performance of train/validation set")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_perf_lstm")
            ]
        )
    ], className = "card border-primary mb-3"
)

card_plot_pred_man_lstm = dbc.Card(
    [
        dbc.CardHeader(html.H5("Manual prediction")),
        dbc.CardBody(
            [
                html.P("Select start day for Prediction:"),
                dcc.DatePickerSingle(id = "datepicker_single_lstm", min_date_allowed = date(1980,1,1), 
                                    max_date_allowed = date.today() - timedelta(days = 17), initial_visible_month = date.today() - timedelta(days = 17), 
                                    date = date.today() - timedelta(days = 17)),
                html.P(),
                dcc.Graph(id = "plot_pred_man_lstm", figure = {})
            ]
        )
    ], className = "card border-primary mb-3"
)

card_perf_pred_man_lstm = dbc.Card(
    [
        dbc.CardHeader(html.H5("Performance manual prediciton")),
        dbc. CardBody(
            [
                html.Div(id = "output_div_perf_pred_man_lstm")
            ]
        )
    ], className = "card border-primary mb-3"
)

layout = dbc.Container(
    [
        dbc.Row([
                html.P("Choose number of days to predict:"),
                dcc.Dropdown(id = "drop_days_lstm", options = [1, 3, 5, 10, 30], value = 10, style = {"color": "black"}),
                html.P()
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_pred_lstm
                ], width = 9
            ),
            dbc.Col(
                [
                    card_table_pred_lstm
                ], width = 3
            )
        ]),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_trainvalsets_lstm
                ], width = 9
            ),
            dbc.Col(
                [
                    card_perf_lstm
                ], width = 3
            )
        ]),
        html.Hr(style = {"margin-top": "-4px"}),
        dbc.Row([
            dbc.Col(
                [
                    card_plot_pred_man_lstm
                ], width = 9
            ),
            dbc.Col(
                [
                    card_perf_pred_man_lstm
                ], width = 3
            )
        ]),
        dcc.Store(id = "forecast_store_lstm"),
    ], fluid = True
)

@callback( #Führe LSTM-Methode aus, speichere Vorhersage im Store, aktualisiere Train/Vald. Plot und das Performance Div
    Output("plot_trainvalsets_lstm", "figure"),
    Output("output_div_perf_lstm", "children"),
    Output("forecast_store_lstm", "data"),
    Input("data_store", "data"),
    Input("ticker_store", "data"),
    Input("token", "value"))
def update_TrainValPlotPerf_StorePred_lstm(data, ticker, token):
    hist_data = pd.read_json(data, orient = "split")
    hist_data.drop(columns = ["Open", "High", "Low", "Volume", "Adj Close"], inplace = True)
    ticker_data = json.loads(ticker)
    ticker_data_dic = dict(ticker_data)
    currency = ticker_data_dic["currency"]
    today = datetime.today().date()
    start = today - timedelta(days=730) #Zeitraum anpassen (2Jahre)
    hist_data = hist_data.loc[hist_data.index >= np.datetime64(start)] #Datensatz filtern
    true_data = hist_data.copy()
    scaler = MinMaxScaler(feature_range = (0, 1))
    hist_data[["Close"]] = scaler.fit_transform(hist_data[["Close"]]) #Daten skalieren
    window_size = 50 #Windowgröße definieren
    last_price = hist_data.iloc[-1][-1] #Letzten Kurs speichern
    data_for_func = hist_data.copy()
    windowed_df = data_to_windowed_data(data_for_func, window_size) #Erhalte Windowed DF
    dates, X, y = windowed_df_to_d_x_y(windowed_df)
    train_split = int(len(dates) * 0.8) #Aufteilung der Daten in Training und Valid.
    dates_train, X_train, y_train = dates[:train_split], X[:train_split], y[:train_split]
    dates_val, X_val, y_val = dates[train_split:], X[train_split:], y[train_split:]

    train_predictions, val_predictions, forecast_df = LSTM(X_train, y_train, X_val, y_val, window_size, last_price, scaler, token) #Prognose

    train_predictions = scaler.inverse_transform(train_predictions) #Skaliere vorhergesagtes Training/Vald. zurück
    val_predictions = scaler.inverse_transform(val_predictions)
    y_train_rescaled = y_train.reshape(len(y_train), 1) #Ändere Format
    y_train_rescaled = scaler.inverse_transform(y_train_rescaled) #Skaliere tatsächliche Trainingsdaten zurück
    average_price_train = y_train_rescaled.mean()
    y_val_rescaled = y_val.reshape(len(y_val), 1)
    y_val_rescaled = scaler.inverse_transform(y_val_rescaled) #Skaliere tatsächliche TValidierungssdaten zurück
    average_price_val = y_val_rescaled.mean()
    #Berechne Kennzahlen
    mae_train = round(mean_absolute_error(y_train_rescaled, train_predictions), 2)
    mae_scaled_train = round((mae_train/average_price_train)*100, 2)
    mse_train = round(mean_squared_error(y_train_rescaled, train_predictions), 2)
    rmse_train = round(math.sqrt(mse_train), 2)
    mae_valid = round(mean_absolute_error(y_val_rescaled, val_predictions), 2)
    mae_scaled_valid = round((mae_valid/average_price_val)*100, 2)
    mse_valid = round(mean_squared_error(y_val_rescaled, val_predictions), 2)
    rmse_valid = round(math.sqrt(mse_valid), 2)
    output = [ #Für Performance Div
        html.P(f"MAE: {mae_train:.2f} (Train) {mae_valid:.2f} (Validation)"),
        html.P(f"MAE Scaled: {mae_scaled_train} % (Train) {mae_scaled_valid} % (Validation)"),
        html.P(f"MSE: {mse_train:.2f} (Train) {mse_valid:.2f} (Validation)"),
        html.P(f"RMSE: {rmse_train} (Train) {rmse_valid} (Validation)")
    ]
    df_valid = pd.DataFrame(index = dates_val, columns = ["True", "Pred"], data = np.hstack((y_val_rescaled, val_predictions))) #Erstellt DF für Train/Vald 
    df_train = pd.DataFrame(index = dates_train, columns = ["True", "Pred"], data = np.hstack((y_train_rescaled, train_predictions)))
    true_data = true_data.iloc[window_size:] #Speichere die tasächlichen Kurse für die prognostizierten Tage ab
    #Erstelle Plot
    fig_trainval_lstm = px.line(template = "simple_white")
    fig_trainval_lstm.add_trace(go.Scatter(x = true_data.index, y = true_data["Close"], mode = "lines", name = "True Data", line_color = "red"))
    fig_trainval_lstm.add_trace(go.Scatter(x = df_train.index, y = df_train["Pred"], mode = "lines", name = "Train Prediction", line_color = "blue"))
    fig_trainval_lstm.add_trace(go.Scatter(x = df_valid.index, y = df_valid["Pred"], mode = "lines", name = "Validation Prediction", line_color = "green"))
    fig_trainval_lstm.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}")
    Speicher die Vorhersage
    forecast = forecast_df["Close"].tolist()
    last_date = hist_data.index[-1]
    market_calendar = get_calendar("NYSE") #nur Börsentage
    next_dates = market_calendar.valid_days(start_date = last_date + timedelta(days = 1), end_date = last_date + timedelta(days = 60))
    next_days_needed = next_dates[:30]
    for i in range(30):
        next_day = next_days_needed[i]
        next_day_date = next_day.date()
        hist_data.loc[np.datetime64(next_day_date)] = forecast[i]
    hist_data = hist_data.iloc[-30:] #nur die vorhergesagten Tage
    return fig_trainval_lstm, output, hist_data.to_json(orient = "split")

@callback( #Update Prognose-Tabelle (wie bei pg3)
    Output("data_table_pred_lstm", "data"),
    Output("data_table_pred_lstm", "columns"),
    Output("data_table_pred_lstm", "style_data_conditional"),
    Output("data_table_pred_lstm", "style_header"),
    Input("forecast_store_lstm", "data"),
    Input("data_store", "data"),
    Input("drop_days_lstm", "value"))
def update_PredTable_lstm(forecast_data, data, count_days):
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

@callback( #Update Prognose Plot (wie bei pg3)
    Output("plot_pred_lstm", "figure"),
    Input("forecast_store_lstm", "data"),
    Input("data_store", "data"),
    Input("drop_days_lstm", "value"),
    Input("ticker_store", "data"))
def update_PlotPred_lstm(forecast_data, data, count_days, ticker):
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
    df_pred_add_last_element = merged_df.tail(count_days + 1)
    fig_pred_lstm = px.line(template = "simple_white")
    fig_pred_lstm.add_trace(go.Scatter(x = hist_data.index, y = hist_data["Close"], mode = "lines", name = "Data", line_color = "blue"))
    fig_pred_lstm.add_trace(go.Scatter(x = df_pred_add_last_element.index, y = df_pred_add_last_element["Close"],mode = "lines", name = "Prediction", line_color = "red"))
    fig_pred_lstm.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}")
    return fig_pred_lstm

@callback( #Update manuelle Prognose (Plot und Performance)
    Output("plot_pred_man_lstm", "figure"),
    Output("output_div_perf_pred_man_lstm", "children"),
    Input("data_store", "data"),
    Input("ticker_store", "data"),
    Input("datepicker_single_lstm", "date"),
    Input("token", "value"))
def update_Pred_Man_lstm(data, ticker, date, token):
    #Vorgehen identisch zu oben
    hist_data = pd.read_json(data, orient = "split")
    date_format = datetime.strptime(date, "%Y-%m-%d").date()
    start = date_format - timedelta(days = 730)
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

    forecast_df = LSTM_man_pred(data_for_func, token, date)

    forecast = forecast_df["Close"].tolist()
    market_calendar = get_calendar("NYSE")
    next_dates = market_calendar.valid_days(start_date = last_date + timedelta(days = 1), end_date = last_date + timedelta(days = 20))
    next_days_needed = next_dates[:10]
    for i in range(10):
        next_day = next_days_needed[i]
        next_day_date = next_day.date()
        hist_data.loc[np.datetime64(next_day_date)] = forecast[i]
    hist_data = hist_data.iloc[-10:]
    fig_man_pred_lstm = px.line(template = "simple_white")
    fig_man_pred_lstm.add_trace(go.Scatter(x = hist_data.index.strftime("%Y-%m-%d"), y = hist_data["Close"], mode = "lines", name = "Prediction", line_color = "red"))
    fig_man_pred_lstm.add_trace(go.Scatter(x = hist_data_for_pred.index.strftime("%Y-%m-%d"), y = hist_data_for_pred["Close"], mode = "lines", name = "True Data", line_color = "blue"))
    fig_man_pred_lstm.update_layout(xaxis_title = "Date", yaxis_title = f"Close Price in {currency}", title = "Prediction", xaxis= {"type": "category"})
    mae = round(mean_absolute_error(hist_data_for_pred["Close"], hist_data["Close"]), 2)
    mae_scaled = round((mae/hist_data_for_pred["Close"].mean())*100, 2)
    mse = round(mean_squared_error(hist_data_for_pred["Close"], hist_data["Close"]), 2)
    rmse = round(math.sqrt(mse) ,2)
    output = [
        html.P(f"Mean Absolute Error: {mae}"),
        html.P(f"Mean Absolute Error Scaled: {mae_scaled} %"),
        html.P(f"Mean Squared Error: {mse}"),
        html.P(f"Rooted Mean Squared Error: {rmse}")
    ]
    return fig_man_pred_lstm, output
