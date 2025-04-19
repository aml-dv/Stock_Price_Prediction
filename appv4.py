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
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests_cache

# Define global constants
time_steps = 60  # Number of time steps for LSTM
DATA_FOLDER = 'stock_data'  # Folder containing stock CSV files
CSV_PATH = 'static/nifty50.csv'  # Path to the CSV with stock metadata

app = Flask(__name__)

# Ensure stock data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

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

class FinancialSentimentAnalyzer:
    def __init__(self, exchange='NS', num_headlines=10):
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        self.exchange = exchange
        self.num_headlines = num_headlines
        self.companies = self._load_companies()
        requests_cache.install_cache('yahoo_cache', expire_after=3600)

    def _load_companies(self, csv_file='name.csv', symbol_col='name'):
        try:
            if not os.path.exists(csv_file):
                return []
            df = pd.read_csv(csv_file)
            if symbol_col not in df.columns:
                return []
            return df[symbol_col].str.strip().str.upper().tolist()
        except Exception:
            return []

    def _construct_url(self, company_symbol):
        return f"https://finance.yahoo.com/quote/{company_symbol}.{self.exchange}/news/"

    def get_headlines(self, company_symbol):
        url = self._construct_url(company_symbol)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = []
            selectors = ['li.js-stream-content h3', 'h3[data-test-locator="headline"]', r'h3.Mb\(5px\)', 'div[data-test-locator="mega"] h3', 'h3']
            for selector in selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        headlines = [h.get_text(strip=True) for h in elements]
                        headlines = [h for h in headlines if len(h.split()) >= 4]
                        if headlines:
                            break
                except Exception:
                    continue
            return headlines[:self.num_headlines] or ["No valid headlines found (minimum 4 words)"]
        except requests.RequestException:
            return [f"Network error"]
        except Exception:
            return [f"Processing error"]

    def analyze_sentiment(self, headlines):
        if not headlines or headlines[0].startswith(("No", "Network", "Processing")):
            return [], "No data", 0.0
        results = []
        compound_scores = []
        for idx, text in enumerate(headlines, 1):
            scores = self.sia.polarity_scores(text)
            if scores['compound'] >= 0.15:
                sentiment = "Strongly Positive"
            elif scores['compound'] >= 0.05:
                sentiment = "Positive"
            elif scores['compound'] <= -0.15:
                sentiment = "Strongly Negative"
            elif scores['compound'] <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            compound_scores.append(scores['compound'])
            results.append({
                'id': idx,
                'text': text,
                'sentiment': sentiment,
                'score': scores['compound']
            })
        avg_score = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0
        overall = "Strongly Positive" if avg_score >= 0.15 else "Positive" if avg_score >= 0.05 else "Strongly Negative" if avg_score <= -0.15 else "Negative" if avg_score <= -0.05 else "Neutral"
        return results, overall, avg_score

@app.route('/')
def display_stocks():
    try:
        df = pd.read_csv(CSV_PATH, usecols=['name', 'code_name', 'logo'])
        stocks = [{
            'name': row['name'].strip(),
            'code': row['code_name'].strip(),
            'logo': row['logo'].strip().replace('\\', '/').replace('static/', '')
        } for _, row in df.iterrows()]
        return render_template('stocks_16.html', stocks=stocks)
    except Exception as e:
        app.logger.error(f"Data loading error: {str(e)}")
        return "Error loading stock list", 500

@app.route('/predict', methods=['POST'])
def predict():
    stock_code = request.form.get('stock_code')
    forecast_days = 70  # Default forecast days

    # Find the corresponding CSV file
    stock_file = next((f for f in os.listdir(DATA_FOLDER) if f.replace('.csv', '').upper() == stock_code.upper()), None)
    if not stock_file:
        return jsonify({'error': 'Stock data not found'}), 400

    # Load and process the stock data
    data = pd.read_csv(os.path.join(DATA_FOLDER, stock_file), thousands=',')
    date_column = next((col for col in ['Date ', 'date ', 'DATE '] if col in data.columns), None)
    if not date_column:
        return jsonify({'error': "No recognizable date column found in CSV"}), 400
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    data.sort_index(inplace=True)  # Ensure data is in chronological order

    if 'series' in data.columns:
        data = data[data['series'] == 'EQ']
    data = data[['close ']].copy()
    if 'close ' not in data.columns:
        return jsonify({'error': "CSV must contain a 'close ' column"}), 400
    data.dropna(inplace=True)

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
    future_dates = [future_start_date + timedelta(days=i) for i in range(forecast_days)]

    # Combine all dates in chronological order
    all_dates = list(train_actual_idx) + list(test_actual_idx) + future_dates
    train_actual_idx = all_dates[:len(train_actual_idx)]
    test_actual_idx = all_dates[len(train_actual_idx):len(train_actual_idx) + len(test_actual_idx)]
    future_dates = all_dates[len(train_actual_idx) + len(test_actual_idx):]

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_actual_idx, y=y_lstm_train_inv, mode='lines', name='Actual Price (Train)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train_actual_idx, y=lstm_train_pred.flatten(), mode='lines', name='LSTM Prediction (Train)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=test_actual_idx, y=y_lstm_test_inv, mode='lines', name='Actual Price (Test)', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=test_actual_idx, y=lstm_test_pred.flatten(), mode='lines', name='LSTM Prediction (Test)', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Future Prediction', line=dict(color='green', dash='dot')))

    fig.update_layout(
        title=f"{stock_code} Stock Price Prediction (LSTM) from {data.index[time_steps].date()} to {data.index[-1].date()} + {forecast_days} Days Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            range=[data.index[time_steps].date(), future_dates[-1]],
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
    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    # Sentiment Analysis
    analyzer = FinancialSentimentAnalyzer(exchange='NS', num_headlines=10)
    company_symbol = stock_code  # Use stock_code as company_symbol
    headlines = analyzer.get_headlines(company_symbol)
    sentiment_results, overall_sentiment, avg_score = analyzer.analyze_sentiment(headlines)

    return render_template('plot.html', plot_html=plot_html, sentiment_results=sentiment_results, overall_sentiment=overall_sentiment, avg_score=avg_score)

if __name__ == '__main__':
    app.run(debug=True)