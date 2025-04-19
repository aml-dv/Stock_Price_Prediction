import yfinance as yf
stock = yf.Ticker('RELIANCE.NS')
hist_data = stock.history(start='2020-01-01', end='2025-04-15')
print(hist_data)