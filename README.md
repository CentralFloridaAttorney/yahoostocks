# yahoostocks

When you create an instance of YahooStock, it will either load the symbol's data from your files or get it from Yahoo

To create an instance with data for Microsoft: msft_stock = YahooStock('MSFT')

To create training and testing splits from the instance: x_train, y_train, x_target, y_target = msft_stock.get_test_train_split(msft_stock.price_frame, 3, 6, .87, 5)

If anyone has any suggestions or requests then let me know!

yahoostocks is not affiliated with the company Yahoo(exclamation)
