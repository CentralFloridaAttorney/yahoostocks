import numpy
import pandas


class Classifier:
    @staticmethod
    def from_threshold(_numpy_column, _threshold_value=1):
        cols_above_threshold = []
        for x in _numpy_column:
            if x > _threshold_value:
                cols_above_threshold.append(1.0)
            else:
                cols_above_threshold.append(0.0)
        return pandas.DataFrame(cols_above_threshold)

    @staticmethod
    def get_average_col_value(_dataframe, _col_num=0):
        col_average = _dataframe[_col_num].mean()
        return col_average

    @staticmethod
    def get_classification_greater_prior(_dataframe, _col_name, _days_before):
        _data = _dataframe
        _data = _data[_col_name]
        _data = pandas.DataFrame(_data)
        _classification = numpy.where((_data > _data.shift(-_days_before)), 1.0, 0.0)
        return _classification
