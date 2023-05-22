import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
import pandas as pd


# Retrieve stocks for the past month
def get_stock_data(stock):
    data = yf.download(stock, period="1mo", interval="1d")
    return data

# Fit ARIMA regression model
def fit_arima(data):
    model = ARIMA(data, order=(1, 0, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return forecast[0]

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1("Stock Price Prediction with ARIMA"),
    html.Div([
        html.Label("Enter stock abbreviation (4 symbols):"),
        dcc.Input(id="input-stock", type="text", value="MSFT"),
        html.Button("Predict", id="btn-predict", n_clicks=0)
    ]),
    dcc.Graph(id="stock-graph"),
    html.Div(id="prediction-output")
])

# Callback function to update the graph and prediction output
@app.callback(
    [Output("stock-graph", "figure"), Output("prediction-output", "children")], 
    [Input("btn-predict", "n_clicks")], 
    [dash.dependencies.State("input-stock", "value")]
)
def update_graph(n_clicks, stock): # n_clicks is the number of times the button has been clicked and stock is the stock abbreviation
    if n_clicks > 0:
        stock_data = get_stock_data(stock)
        prediction = fit_arima(stock_data["closing price"])

        # Create plotly trace for stock data
        stock_trace = go.Scatter(
            x=stock_data.index,
            y=stock_data["closing price"],
            mode='lines',
            name='Stock Data'
        )

        # Create plotly trace for predicted value
        prediction_trace = go.Scatter(
            x=[stock_data.index[-1], stock_data.index[-1] + pd.DateOffset(days=1)],
            y=[stock_data["closing price"].iloc[-1], prediction],
            mode='lines',
            name='Prediction'
        )

        # Create plotly layout
        layout = go.Layout(
            title=f"Stock Price Evolution: {stock}",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price (USD)')
        )

        # Create plotly figure by combining the traces and layout
        fig = go.Figure(data=[stock_trace, prediction_trace], layout=layout)

        return fig, f"Predicted stock price for the next day: {prediction:.2f} USD"
    else:
        return {}, ""

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
