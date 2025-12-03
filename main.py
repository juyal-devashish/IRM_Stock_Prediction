


### COLORED MNIST






### YAHOO FINANCE

import yfinance as yf

### Stock price history
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y")
print(data)

### Downloading multiple tickers
df = yf.download(["AAPL", "MSFT", "GOOG"], period="6mo")
print(df)

### Get info
ticker = yf.Ticker("TSLA")
print(ticker.info)            # Company info
print(ticker.financials)      # Income statements
print(ticker.balance_sheet)   # Balance sheet
print(ticker.cashflow)        # Cash flow


# 1. Frame the finance data in terms of how we want the environments setup

# First will have to look how they do it from the paper


# Need to do 80/10/10 data split