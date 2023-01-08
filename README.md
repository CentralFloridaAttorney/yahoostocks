# yahoostocks
yahoostocks is not affiliated with Yahoo

yahoostocks is a simple tool to make training and testing splits using stock market data from Yahoo

When you instant the class object, it will either load the stock data from your files or get it from Yahoo

To create an instance with stock data for Microsoft: msft_stock = YahooStock('MSFT')

To create training and testing splits: x_train, y_train, x_test, y_test = msft_stock.get_test_train_split(msft_stock.price_frame, 3, 6, .87, 5)
