
import yfinance as yf

interval = "1m"
symbol='RAMCOIND.NS'
stock_data = yf.download(symbol,period="1d",interval=interval,group_by="ticker", prepost=True, proxy=None, rounding=True,  auto_adjust=False, back_adjust=False,actions=True, threads=True, timeout=None)

print(stock_data)
stock_data.to_csv("stock_data1.csv", index=True)