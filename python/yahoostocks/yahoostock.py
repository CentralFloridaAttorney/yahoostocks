import json
import math

import numpy
import pandas
import torchvision.transforms as transforms

from python.yahoostocks.classifier import Classifier
from python.yahoostocks.stock_data_getter import StockData

tickerX = 'HRL'
BASE_DIR = '../../data/stocks/'
START_DATE = '1900-01-01'
END_DATE = '2022-06-02'


class YahooStock:
    def __init__(self, init_ticker="F"):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.training_set = numpy.empty_like
        self.validation_set = numpy.empty_like
        self.validation_loader = None
        self.training_loader = None
        self.ticker = init_ticker
        try:
            text_file = open(BASE_DIR + self.ticker + '.json', 'r')
            self.text_string = text_file.read()
            text_file.close()
        except FileNotFoundError:
            self.load_ticker()
        except ValueError:
            self.load_ticker()
        except Exception as err:
            print('jsonstock.__init__ error: ' + str(err))

        self.text_string = self.text_string.replace('\n', '')
        self.json_data = json.loads(self.text_string)
        prices_a = self.json_data[self.ticker]['prices']
        these_prices = []
        for price in prices_a:
            these_prices.append(price)
        self.price_frame = pandas.DataFrame(these_prices)
        self.data_frame = self.price_frame.to_numpy()

    def load_ticker(self):
        yf = StockData(self.ticker)
        self.text_string = yf.get_past_data(start_date=START_DATE, end_date=END_DATE, time_interval='daily')
        print('WARNING: Loaded data directly from yahoo')
        self.text_string = json.dumps(self.text_string)
        text_file = open(BASE_DIR + self.ticker + '.json', 'w+')
        text_file.write(self.text_string)
        text_file.close()

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
        """
        This method returns a list where the value is 1 when price_data exceeds the threshold_value.  The values in the list correspond to the order of the values in price_frame.

        Parameters
        ----------
        data_column : DataFrame
            The data column to be added to file_data
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
        return Classifier().from_threshold(data_column, threshold_value)

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
        _extra_rows_1 = (_num_rows) % (_batch_size * _train_ratio)
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
        _y_train = _data[_num_x:(_num_x + _num_y), 0:(_split_col)]
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

    tickers = ['MSFT', 'SAP']
    for ticker in tickers:
        YahooStock(ticker)

    stockItem.drop_column(7)
    col1 = stockItem.get_price_data(4)
    # classification = stockItem.get_classification_greater_prior(2, 7)
    classification = Classifier.get_classification_greater_prior(_dataframe=stockItem.price_frame, _col_name='high',
                                                                 _days_before=2)
    stockItem.add_column(classification)
    data = stockItem.price_frame.to_numpy()
    x_train, y_train, x_test, y_test = stockItem.get_test_train_split(data, 3, 7, .87)
    # the following code returns a list where the value is 1 when the price data exceeds the threshold_value
    average_value = Classifier.get_average_col_value(_dataframe=col1, _col_num=0)
    indices_above_threshold = Classifier.from_threshold(_numpy_column=col1,
                                                        _threshold_value=(average_value + (average_value * .2)))
    print('Finished!')
