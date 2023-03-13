from python.yahoostocks.yahoostocks import YahooStocks
TICKER = 'GM'


def test_init():
    stock_item = YahooStocks(TICKER)
    print('test_init: ' + stock_item.ticker_symbol)


def test_init_dict(_stock_dict):
    stock_array = []
    for ticker in _stock_dict:
        this_stock = YahooStocks(ticker)
        stock_array.append(this_stock)
    print('test_init_array number of stocks: ' + str(len(stock_array)))


def test_get_column():
    stock_item = YahooStocks(TICKER)
    columns = stock_item.price_frame.columns
    price_data_column = stock_item.get_column(columns[2])
    print('test_get_column: ' + str(price_data_column.size))


def test_get_test_train_split():
    stock_object = YahooStocks(TICKER)
    x_train, y_train, x_target, y_target = stock_object.get_test_train_split(_data=stock_object.price_frame,
                                                                             _train_end_col=3, _batch_size=6,
                                                                             _train_ratio=.87, _target_column_start=5)
    print('test_get_test_train_split: ' + y_train.columns[0])


def test_get_classification_greater_prior():
    stock_object = YahooStocks(TICKER)
    classification = stock_object.get_classification_greater_prior(2, 4)
    print('get_classification_greater_prior: ' + str(classification))



if __name__ == '__main__':
    test_init()
    stock_dict = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "USDT-USD": "Tether",
        "SNY": "Sanofi-Aventis",
        "NVS": "Novartis",
        "TSLA": "Tesla",
        "AXP": "American express",
        "BLK": "Blackrock",
        "BABA": "Alibaba",
        "NVDA": "Nvidia",
        "PYPL": "Paypal"
    }
    test_init_dict(stock_dict)
    test_get_column()
    test_get_test_train_split()
    test_get_classification_greater_prior()
    print('Finished YahooStock: ' + TICKER)
