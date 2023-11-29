import yfinance as yf
# Import talib library
import talib

# Fetch daily stock price data of Apple
aapl_stock_data = yf.download('AAPL','2015-01-01','2021-07-28')
aapl_stock_data.head()