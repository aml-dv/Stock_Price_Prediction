from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from datetime import timedelta

# Define global constants
time_steps = 60  # Number of time steps for LSTM
DATA_FOLDER = 'stock_data'  # Folder containing CSV files

app = Flask(__name__)

# Ensure data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Define stock metadata (code, name, logo)
stocks = [
    {'code': 'TATAMOTORS', 'name': 'Tata Motors Ltd.', 'logo': 'logos/tatamotors.png'},
    # Add more stocks as needed, e.g.,
    # {'code': 'RELIANCE', 'name': 'Reliance Industries Ltd.', 'logo': 'logos/reliance.png'},
]

# Data preparation and model training functions
def create_lstm_dataset(data, time_steps=60):
    X, y = [], []
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close ']])
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def forecast_future_prices(model, last_sequence, forecast_days, scaler):
    future_preds = []
    current_sequence = last_sequence.copy()
    for _ in range(forecast_days):
        current_sequence_reshaped = current_sequence.reshape((1, time_steps, 1))
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        future_preds.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

@app.route('/', methods=['GET'])
def stock_selection():
    return render_template('stocks.html', stocks=stocks)

@app.route('/predict', methods=['POST'])
def predict():
    stock_code = request.form.get('stock_code')
    forecast_days = int(request.form.get('forecast_days', 70))

    # Validate stock code
    if not stock_code or stock_code.upper() not in [s['code'].upper() for s in stocks]:
        return jsonify({'error': 'Invalid stock code'}), 400

    # Load CSV file for the selected stock
    stock_file = os.path.join(DATA_FOLDER, f"{stock_code.upper()}.csv")
    if not os.path.exists(stock_file):
        return jsonify({'error': f'CSV file for {stock_code} not found'}), 400

    # Load data, handling commas as thousand separators
    data = pd.read_csv(stock_file, thousands=',')
    print("Available columns:", data.columns.tolist())  # Debug: Print all column names

    # Identify and set the date column as index
    date_column = next((col for col in ['Date ', 'date ', 'DATE '] if col in data.columns), None)
    if not date_column:
        raise ValueError("No recognizable date column found. Check for 'Date', 'date', or 'DATE' in the CSV.")
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)

    # Check if 'series' column exists before filtering
    if 'series' in data.columns:
        data = data[data['series'] == 'EQ']  # Filter for equity series
    else:
        print("Warning: 'series' column not found. Skipping equity series filtering.")

    data = data[['close ']].copy()  # Use CLOSE column
    if 'close ' not in data.columns:
        raise ValueError("CSV must contain a 'close ' column")
    data.dropna(inplace=True)

    print(f"Loaded data for {stock_code} from {data.index[0].date()} to {data.index[-1].date()} with {len(data)} trading days")

    # Prepare data for LSTM
    X_lstm, y_lstm, scaler = create_lstm_dataset(data, time_steps)
    if len(X_lstm) == 0:
        return jsonify({'error': 'Not enough data after time_steps'}), 400

    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    # Split data
    train_size = int(len(X_lstm) * 0.8)
    X_lstm_train, X_lstm_test = X_lstm[:train_size], X_lstm[train_size:]
    y_lstm_train, y_lstm_test = y_lstm[:train_size], y_lstm[train_size:]

    # Build and train LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32, verbose=0)

    # Get LSTM predictions for train and test
    lstm_train_pred = lstm_model.predict(X_lstm_train, verbose=0)
    lstm_test_pred = lstm_model.predict(X_lstm_test, verbose=0)

    # Inverse transform predictions
    lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
    lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
    y_lstm_train_inv = scaler.inverse_transform([y_lstm_train])[0]
    y_lstm_test_inv = scaler.inverse_transform([y_lstm_test])[0]

    # Forecast future prices
    last_sequence = scaler.transform(data[['close ']].iloc[-time_steps:]).flatten()
    future_preds = forecast_future_prices(lstm_model, last_sequence, forecast_days, scaler)

    # Define indices for plot
    train_actual_idx = data.index[time_steps:train_size + time_steps]
    test_actual_idx = data.index[train_size + time_steps:train_size + time_steps + len(y_lstm_test)]
    future_start_date = data.index[-1] + timedelta(days=1)
    future_dates = sorted([future_start_date + timedelta(days=i) for i in range(forecast_days)])

    # Create plot
    fig = go.Figure()

    # Training actual prices
    fig.add_trace(go.Scatter(
        x=train_actual_idx,
        y=y_lstm_train_inv,
        mode='lines',
        name='Actual Price (Train)',
        line=dict(color='blue')
    ))

    # Training predictions
    fig.add_trace(go.Scatter(
        x=train_actual_idx,
        y=lstm_train_pred.flatten(),
        mode='lines',
        name='LSTM Prediction (Train)',
        line=dict(color='orange')
    ))

    # Testing actual prices
    fig.add_trace(go.Scatter(
        x=test_actual_idx,
        y=y_lstm_test_inv,
        mode='lines',
        name='Actual Price (Test)',
        line=dict(color='blue', dash='dash')
    ))

    # Testing predictions
    fig.add_trace(go.Scatter(
        x=test_actual_idx,
        y=lstm_test_pred.flatten(),
        mode='lines',
        name='LSTM Prediction (Test)',
        line=dict(color='orange', dash='dash')
    ))

    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode='lines',
        name='Future Prediction',
        line=dict(color='green', dash='dot')
    ))

    # Update layout
    fig.update_layout(
        title=f"{stock_code} Stock Price Prediction (LSTM) from {data.index[time_steps].date()} to {data.index[-1].date()} + {forecast_days} Days Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            range=[data.index[time_steps].date(), future_dates[-1] + timedelta(days=10)],
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        ),
        template="plotly_white"
    )

    # Save plot to HTML
    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    return render_template('plot.html', plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)