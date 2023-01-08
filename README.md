# yahoostocks
yahoostocks is not affiliated with the company Yahoo(exclamation)

yahoostocks is a simple tool to make training and testing splits using stock market data from Yahoo

When you instance of YahooStock, it will either load the stock data from your files or get it from Yahoo

For example, to create an instance with stock data for Microsoft use the following syntax: msft_stock = YahooStock('MSFT')

For example, to create training and testing splits for msft_stock use the following syntax: x_train, y_train, x_test, y_test = msft_stock.get_test_train_split(msft_stock.price_frame, 3, 6, .87, 5)
