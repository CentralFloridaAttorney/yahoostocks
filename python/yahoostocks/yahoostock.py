import json
import math
import os

import numpy
import pandas
import pkg_resources
from pkg_resources import resource_listdir

from python.yahoostocks.yahoofinance import YahooFinancials
from python.yahoostocks.classifier import Classifier

# BASE_PATH = '../../data/stocks/'
#BASE_PATH = os.path.dirname(__file__)

symbol_dict = {
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


class YahooStock:
    def __init__(self, init_ticker="F", _stock_dict=None):
        """
        YahooStock gets stock data from Yahoo and creates testing and training sets

        :param init_ticker: the ticker symbol of a publicly traded stock
        """
        self.ticker_file_path = pkg_resources.resource_filename("data", "stocks/" +init_ticker + ".json")

        self.these_prices = []

        self.ticker = init_ticker
        if _stock_dict is not None:
            self.stock_dict = _stock_dict
            for ticker in self.stock_dict:
                this_stock = YahooStock(ticker)
                this_price_frame = this_stock.price_frame
                self.these_prices.append(this_price_frame)

        try:
            # 'data/stocks/AMD.json'
            text_file = open(self.ticker_file_path, 'r')
            self.text_string = text_file.read()
            text_file.close()
        except FileNotFoundError:
            self._load_ticker()
        except ValueError:
            self._load_ticker()

        self.text_string = self.text_string.replace('\n', '')
        self.json_data = json.loads(self.text_string)
        prices_a = self.json_data[self.ticker]['prices']
        these_prices = []
        for price in prices_a:
            these_prices.append(price)
        self.price_frame = pandas.DataFrame(these_prices)
        self.data_frame = self.price_frame.to_numpy()

    def _add_column(self, _data_column):
        _data_column = pandas.DataFrame(_data_column).to_numpy()
        self.data_frame = numpy.concatenate((self.data_frame, _data_column), axis=1)
        return self.data_frame

    def _drop_column(self, _col_num):
        self.data_frame = pandas.DataFrame(self.data_frame).to_numpy()
        self.data_frame = self.data_frame[:, 0:_col_num]
        return self.data_frame

    def _load_ticker(self):
        yf = YahooFinancials(self.ticker)
        self.text_string = yf.get_historical_price_data(start_date='1900-01-01', end_date='2022-06-02',
                                                        time_interval='daily')
        print('WARNING: Loaded data directly from yahoo: ' + self.ticker)
        self.text_string = json.dumps(self.text_string)
        text_file = open(self.ticker_file_path, 'w')
        text_file.write(self.text_string)
        text_file.close()

    def _reset_data_frame(self):
        self.data_frame = self.price_frame

    @staticmethod
    def from_threshold(_numpy_column, _threshold_value=1):
        cols_above_threshold = []
        for x in _numpy_column:
            if x > _threshold_value:
                cols_above_threshold.append(1.0)
            else:
                cols_above_threshold.append(0.0)
        return pandas.DataFrame(cols_above_threshold)

    def get_classification_greater_prior(self, _col_num, _days_before):
        """
        This method uses the data stored in self.price_frame
        :param _col_num:
        :param _days_before:
        :return:
        """
        _data = pandas.DataFrame(self.price_frame).to_numpy()
        _data = _data[:, _col_num]
        _data = pandas.DataFrame(_data)
        classified = numpy.where((_data > _data.shift(-_days_before)), 1.0, 0.0)
        return pandas.DataFrame(classified)

    def get_column(self, _col_name):
        """

        :param column_name: the string name of the column (ex. 'Volume')
        :return:
        """
        try:
            temp_v = self.price_frame[_col_name].iloc[:, ]
        except (ValueError, KeyError):
            temp_v = pandas.DataFrame(self.price_frame).to_numpy()
            temp_v = temp_v[:, _col_name]
            temp_v = pandas.DataFrame(temp_v).to_numpy()
        new_dataframe = pandas.DataFrame(temp_v)
        new_dataframe.columns = [_col_name]
        return new_dataframe

    @staticmethod
    def get_test_train_split(_data, _train_start_col=1, _train_end_col=2, _batch_size=4, _train_ratio=0.6, _target_column_start=3, _target_col_end=3):
        """

        :param _data: a pandas.DataFrame with columns to be split for training and testing sets
        :param _train_end_col: the column in the dataframe to divide between x and y values
        :param _batch_size: the number of samples you want in each batch.  This number is used to determine whether rows need to be truncated to make each batch an equal size.
        :param _train_ratio: the percentage of items for training and testing sets
        :param _target_column_start: the index, which starts at 0, of the column that contains the output result.  note that this may be changed to allow a range of output columns

        :return: x_training, y_training, x_target, y_target
        """
        # _batch_size and _train_ratio must be float types
        split_col = int(_train_end_col)
        batch_size = int(_batch_size)

        num_rows, num_cols = _data.shape

        extra_rows_1 = num_rows % (_batch_size * _train_ratio)
        extra_rows_2 = (num_rows % _batch_size) * _train_ratio
        extra_rows_3 = (num_rows * _train_ratio) % _batch_size

        num_x = int(math.floor(num_rows * _train_ratio))
        extra_rows_x = num_x % _batch_size
        num_x = int(num_x - extra_rows_x)
        num_y = num_rows - num_x
        extra_rows_y = num_y % _batch_size
        num_y = num_y - extra_rows_y
        x_training = _data.iloc[0:num_x, _train_start_col:_train_end_col]
        x_target = _data.iloc[0:num_x, _target_column_start:_target_col_end]
        y_training = _data.iloc[num_x:(num_x + num_y), _train_start_col:_train_end_col]
        y_target = _data.iloc[num_x:(num_x + num_y), _target_column_start:_target_col_end]
        return x_training, y_training, x_target, y_target  # , _x_target, _y_target


if __name__ == '__main__':
    TICKER = "MMM"
    stock_object = YahooStock(TICKER)
    col_5 = stock_object.get_column(4)
    classification = stock_object.get_classification_greater_prior(2, 4)
    x_train, y_train, x_target, y_target = stock_object.get_test_train_split(_data=stock_object.price_frame,
                                                                             _train_end_col=3, _batch_size=6,
                                                                             _train_ratio=.87, _target_column_start=5)
    # stock_objects = YahooStock(symbol_dict)
    print('Finished YahooStock: ' + TICKER)
