import json
import math
import urllib.request
from urllib.error import URLError

import numpy
import pandas
import torchvision.transforms as transforms

from python.yahoof.yahoofinance import YahooFinancials
from python.yahoostocks.classifier import Classifier

tickerX = 'LIC'

BASE_DIR = '../../data/stocks/'


class YahooStock:
    def __init__(self, init_ticker="F"):
        """
        YahooStock is a data object.  This implementation loads stock market data for a publicly traded company and makes the data available for use in machine learning projects.
        :param init_ticker: stock market ticker symbol for a publicly traded company
        """
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.training_set = numpy.empty_like
        self.validation_set = numpy.empty_like
        self.validation_loader = None
        self.training_loader = None
        self.ticker = init_ticker
        self.text_string = self.load_ticker()
        self.json_data = self._load_json_data()
        prices_a = self.json_data[self.ticker]['prices']
        these_prices = []
        for price in prices_a:
            these_prices.append(price)
        self.price_frame = pandas.DataFrame(these_prices)
        self.data_frame = self.price_frame.to_numpy()

    def load_ticker(self):
        # if self.ticker.json exists then load the file
        yf = YahooFinancials(self.ticker)
        try:
            # stock_file = open(BASE_DIR+self.ticker+'.json', 'r+')
            # file_string = stock_file.read()
            # file_string = file_string.replace('\n', ' ')
            uh = urllib.request.urlopen('file:' + BASE_DIR + self.ticker + '.json')
            this_data = uh.read()

            print('Retrieved', len(this_data), 'characters')
            text_string = 'default'
            text_string = str(json.loads(this_data.decode("utf-8")))
            text_data = pandas.DataFrame(text_string)

            # text_string = json.loads(file_string.de)
            print('*** Loaded stock data from file: ' + self.ticker)
        except URLError:
            stock_dict = yf.get_historical_price_data(start_date='1900-01-01', end_date='2022-06-02',
                                                      time_interval='daily')
            text_data = pandas.DataFrame(stock_dict)

            text_file = open(BASE_DIR + self.ticker + '.json', 'w')
            text_string = str(stock_dict)
            text_string = text_string.replace('\n', ' ')
            text_file.write(text_string)
            text_file.close()
            print('WARNING: Loaded stock data directly from yahoo: ' + self.ticker)
        # text_string = json.dumps(text_string)
        print('*** finished loading stock data: ' + self.ticker)
        return text_string

    def _load_json_data(self):
        # the file to be converted to
        # json format
        filename = BASE_DIR + self.ticker + '.json'

        # dictionary where the lines from
        # text will be stored
        json_dict = {}

        # creating dictionary
        with open(filename) as fh:
            for line in fh:
                # reads each line and trims of extra the spaces
                # and gives only the valid words
                command, description = line.strip().split(None, 1)

                json_dict[command] = description.strip()

        # creating json file
        # the JSON file is named as test1
        out_file = open("test1.json", "w")
        json.dump(json_dict, out_file, indent=4, sort_keys=False)
        out_file.close()
        return json_dict

    def get_price_data(self, column_name):
        try:
            temp_v = self.price_frame[column_name].iloc[:, ]
        except (ValueError, KeyError):
            temp_v = pandas.DataFrame(self.price_frame).to_numpy()
            temp_v = temp_v[:, column_name]
            temp_v = pandas.DataFrame(temp_v).to_numpy()
        return temp_v

    def add_column(self, _data_column):
        _data_column = pandas.DataFrame(_data_column).to_numpy()
        self.data_frame = numpy.concatenate((self.data_frame, _data_column), axis=1)
        return self.data_frame

    def drop_column(self, _col_num):
        self.data_frame = pandas.DataFrame(self.data_frame).to_numpy()
        self.data_frame = self.data_frame[:, 0:_col_num]
        return self.data_frame

    def reset_data_frame(self):
        self.data_frame = self.price_frame

    @staticmethod
    def get_classification(data_column, threshold_value=0, lagging_average=None):
        """Returns 1.0 in a column when column_data exceeds the threshold_value, if the value is below threshold_value
        then 0.0 is added

        Parameters
        ----------
        data_column : DataFrame
            The options column to be added to file_data
        threshold_value : float
            The minimum value for a true result

        Returns
        -------
        list
            file_data
            :param data_column:
            :param threshold_value:
            :param lagging_average:
        """
        return Classifier.from_threshold(data_column, threshold_value)

    @staticmethod
    def get_test_train_split(_data, _split_col, _batch_size, _train_ratio):
        # _batch_size and _train_ratio must be float types
        _split_col = int(_split_col)
        _batch_size = int(_batch_size)

        try:
            _num_rows, _num_cols = _data.shape
        except TypeError:
            _data = numpy.array(_data)
            _num_rows, _num_cols = _data.shape
        _extra_rows_1 = _num_rows % (_batch_size * _train_ratio)
        _extra_rows_2 = (_num_rows % _batch_size) * _train_ratio
        _extra_rows_3 = (_num_rows * _train_ratio) % _batch_size

        _num_x = int(math.floor(_num_rows * _train_ratio))
        _extra_rows_x = _num_x % _batch_size
        _num_x = int(_num_x - _extra_rows_x)
        _num_y = _num_rows - _num_x
        _extra_rows_y = _num_y % _batch_size
        _num_y = _num_y - _extra_rows_y
        _x_train = _data[0:_num_x, 0:_split_col]
        _x_target = _data[0:_num_x, _split_col:_split_col]
        _x_target = numpy.resize(_x_target, [_num_x, 1])
        _y_train = _data[_num_x:(_num_x + _num_y), 0:_split_col]
        _y_target = _data[_num_x:(_num_x + _num_y), _split_col:_split_col]
        _y_target = numpy.resize(_y_target, [_num_y, 1])
        return _x_train, _x_target, _y_train, _y_target  # , _x_target, _y_target

    def get_classification_greater_prior(self, _col_num, _days_before):
        _data = pandas.DataFrame(self.price_frame).to_numpy()
        _data = _data[:, _col_num]
        _data = pandas.DataFrame(_data)
        _calssification = numpy.where((_data > _data.shift(-_days_before)), 1.0, 0.0)
        return _calssification


if __name__ == '__main__':
    stockItem = YahooStock(tickerX)
    multi_tickers = []
    tickers = ['LIC']
    for ticker in tickers:
        multi_tickers.append(YahooStock(ticker))

    stockItem.drop_column(7)
    col1 = stockItem.get_price_data(4)
    classification = stockItem.get_classification_greater_prior(2, 7)
    stockItem.add_column(classification)
    data = stockItem.price_frame.to_numpy()
    x_train, y_train, x_test, y_test = stockItem.get_test_train_split(data, 3, 7, .87)
    print('Finished!')