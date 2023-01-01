import pandas

from python.yahoostocks.yahoostock import YahooStock

filepath = '../data/csv/nasdaq-listed.csv'

nasdaq_data = pandas.read_csv(filepath)

tickers = nasdaq_data['Symbol'].values

for row in range(77, 79):
    YahooStock(tickers[row])
    print(tickers[row])

print('finished ticker tool')
# print(tickers)
