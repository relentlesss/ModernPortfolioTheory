import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials


# Import stock prices of top ten U.S. capitalization from yahoo finance
assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA','BRK-A','FB','V','UNH']
yahoo_financials = YahooFinancials(assets)
data = yahoo_financials.get_historical_price_data(start_date='2012-06-01', 
                                                end_date='2021-12-31', 
                                                time_interval='monthly')
# Transform yahoo data to pandas dataframe
prices_df = pd.DataFrame({
    a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in assets
    })
    
# Transform pandas dataframe into csv file
data_csv = prices_df.to_csv('stock_price_data.csv')