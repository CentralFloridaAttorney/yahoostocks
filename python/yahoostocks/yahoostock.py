import json
import math

import numpy
import pandas
import torchvision.transforms as transforms

from python.yahoof.yahoofinance import YahooFinancials
from python.yahoostocks.classifier import Classifier
tickerX = 'NIO'
BASE_PATH = '../../data/stocks/'


class YahooStock:
    def __init__(self, init_ticker="F"):
        """
        YahooStock gets stock data from Yahoo and creates testing and training sets

        :param init_ticker: the ticker symbol of a publicly traded stock
        """
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.training_set = numpy.empty_like
        self.validation_set = numpy.empty_like
        self.validation_loader = None
        self.training_loader = None
        self.ticker = init_ticker
        try:
            # 'data/stocks/AMD.json'
            text_file = open(BASE_PATH + self.ticker + '.json', 'r')
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
        text_file = open(BASE_PATH + self.ticker + '.json', 'w')
        text_file.write(self.text_string)
        text_file.close()

    def _reset_data_frame(self):
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
        return Classifier().from_threshold(data_column, threshold_value)

    def get_classification_greater_prior(self, _col_num, _days_before):
        _data = pandas.DataFrame(self.price_frame).to_numpy()
        _data = _data[:, _col_num]
        _data = pandas.DataFrame(_data)
        classified = numpy.where((_data > _data.shift(-_days_before)), 1.0, 0.0)
        return classified

    def get_price_data(self, column_name):
        try:
            temp_v = self.price_frame[column_name].iloc[:, ]
        except (ValueError, KeyError):
            temp_v = pandas.DataFrame(self.price_frame).to_numpy()
            temp_v = temp_v[:, column_name]
            temp_v = pandas.DataFrame(temp_v).to_numpy()
        return temp_v

    @staticmethod
    def get_test_train_split(_data, _train_start_col=2, _batch_size=4, _train_ratio=0.6, _target_column_start=3):
        """

        :param _data: a pandas.DataFrame with columns to be split for training and testing sets
        :param _train_start_col: the column in the dataframe to divide between x and y values
        :param _batch_size: the number of samples you want in each batch.  This number is used to determine whether rows need to be truncated to make each batch an equal size.
        :param _train_ratio: the percentage of items for training and testing sets
        :param _target_column_start: the index, which starts at 0, of the column that contains the output result.  note that this may be changed to allow a range of output columns

        :return: _x_train, _x_target, _y_train, _y_target
        """
        # _batch_size and _train_ratio must be float types
        split_col = int(_train_start_col)
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
        x_training = _data.iloc[0:num_x, 0:_train_start_col]
        x_target = _data.iloc[0:num_x, _target_column_start:_target_column_start+1]
        y_training = _data.iloc[num_x:(num_x + num_y), 0:_train_start_col]
        y_target = _data.iloc[num_x:(num_x + num_y), _target_column_start:_target_column_start+1]
        return x_training, y_training, x_target, y_target  # , _x_target, _y_target


if __name__ == '__main__':
    stock_object = YahooStock(tickerX)
    col_5 = stock_object.get_price_data(4)
    classification = stock_object.get_classification_greater_prior(2, 4)
    x_train, y_train, x_target, y_target = stock_object.get_test_train_split(
        _data=stock_object.price_frame,
        _train_start_col=3,
        _batch_size=6,
        _train_ratio=.87,
        _target_column_start=5
    )
    print('Finished YahooStock: ' + tickerX)
