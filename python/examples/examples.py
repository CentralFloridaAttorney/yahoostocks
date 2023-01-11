from python.yahoostocks.yahoostock import YahooStock
TICKER = 'GM'


def test_init():
    stock_item = YahooStock(TICKER)
    print('test_init: ' + stock_item.ticker)


def test_get_column():
    stock_item = YahooStock(TICKER)
    columns = stock_item.price_frame.columns
    price_data_column = stock_item.get_column(columns[2])
    print('test_get_column: ' + str(price_data_column.size))


def test_get_test_train_split():
    stock_object = YahooStock(TICKER)
    classification = stock_object.get_classification_greater_prior(2, 4)
    x_train, y_train, x_target, y_target = stock_object.get_test_train_split(_data=stock_object.price_frame,
                                                                             _train_end_col=3, _batch_size=6,
                                                                             _train_ratio=.87, _target_column_start=5)
    print('test_get_test_train_split: ' + y_train.columns[0])


if __name__ == '__main__':
    test_init()
    test_get_column()
    test_get_test_train_split()
    print('Finished YahooStock: ' + TICKER)
