# yahoostocks
yahoostocks is not affiliated with the company Yahoo(exclamation)

yahoostocks is a simple tool to make training and testing splits using stock market data from Yahoo using a ticker symbol

When you create an instance of YahooStock, it will either load the symbol's data from your files or get it from Yahoo

To create an instance with data for Microsoft: msft_stock = YahooStock('MSFT')

To create training and testing splits from the instance: x_train, y_train, x_test, y_test = msft_stock.get_test_train_split(msft_stock.price_frame, 3, 6, .87, 5)
