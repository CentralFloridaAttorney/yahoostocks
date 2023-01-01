import calendar
import datetime
import random
import sys
import time
from json import loads

from urllib.request import FancyURLopener


class UrlOpener(FancyURLopener):
    version = 'w3m/0.5.3+git20180125'


class StockDataMain(object):
    def __init__(self, ticker):
        self.ticker = ticker.upper() if isinstance(ticker, str) else [t.upper() for t in ticker]
        self._cache = {}

    DATA_TYPES_FINANCIAL = {
        'income': ['financials', 'incomeStatementHistory', 'incomeStatementHistoryQuarterly'],
        'balance': ['balance-sheet', 'balanceSheetHistory', 'balanceSheetHistoryQuarterly', 'balanceSheetStatements'],
        'cash': ['cash-flow', 'cashflowStatementHistory', 'cashflowStatementHistoryQuarterly', 'cashflowStatements'],
        'keystats': ['key-statistics'],
        'history': ['history']
    }

    _INTERVAL_DICT = {
        'daily': '1d',
        'weekly': '1wk',
        'monthly': '1mo'
    }

    _BASE_YAHOO_URL = 'https://finance.yahoo.com/quote/'


class StockData(StockDataMain):
    def _clean_stock_data(self, hist_data, last_attempt=False):
        data = {}
        for k, v in hist_data.items():
            if k == 'eventsData':
                self._events_data()
            elif 'date' in k.lower():
               self._date_in_lower()
            elif isinstance(v, list):
               self._is_instance()
            else:
                dict_ent = {k: v}
            data.update(dict_ent)
        return data

    def _get_dict(self, up_ticker, statement_type, report_name, hist_obj):
        YAHOO_URL = self._load_yahoo_data(up_ticker, hist_obj)
        cleaned_re_data = self._recursive_data_request(hist_obj, up_ticker)
        dict_ent = {up_ticker: cleaned_re_data}
        return dict_ent

    def _date_in_lower(self):
        if v is not None:
            cleaned_date = self.format_date(v)
            dict_ent = {k: {'formatted_date': cleaned_date, 'date': v}}
        else:
            if last_attempt is False:
                return None
            else:
                dict_ent = {k: {'formatted_date': None, 'date': v}}

    @staticmethod
    def _encode_ticker(ticker_str):
        encoded_ticker = ticker_str.replace('=', '%3D')
        return encoded_ticker

    def _events_data(self):
        event_obj = {}
        if isinstance(v, list):
            dict_ent = {k: event_obj}
        else:
            for type_key, type_obj in v.items():
                formatted_type_obj = {}
                for date_key, date_obj in type_obj.items():
                    formatted_date_key = self.format_date(int(date_key))
                    cleaned_date = self.format_date(int(date_obj['date']))
                    date_obj.update({'formatted_date': cleaned_date})
                    formatted_type_obj.update({formatted_date_key: date_obj})
                event_obj.update({type_key: formatted_type_obj})

    def _get_clean_data(self, api_url):
        raw_data = self._process_load_stock_data(api_url)
        ret_obj = {}
        ret_obj.update({'eventsData': []})
        if raw_data is None:
            return ret_obj
        results = raw_data['chart']['result']
        if results is None:
            return ret_obj
        for result in results:
            tz_sub_dict = {}
            ret_obj.update({'eventsData': result.get('events', {})})
            ret_obj.update({'firstTradeDate': result['meta'].get('firstTradeDate', 'NA')})
            ret_obj.update({'currency': result['meta'].get('currency', 'NA')})
            ret_obj.update({'instrumentType': result['meta'].get('instrumentType', 'NA')})
            tz_sub_dict.update({'gmtOffset': result['meta']['gmtoffset']})
            ret_obj.update({'timeZone': tz_sub_dict})
            timestamp_list = result['timestamp']
            high_price_list = result['indicators']['quote'][0]['high']
            low_price_list = result['indicators']['quote'][0]['low']
            open_price_list = result['indicators']['quote'][0]['open']
            close_price_list = result['indicators']['quote'][0]['close']
            volume_list = result['indicators']['quote'][0]['volume']
            adj_close_list = result['indicators']['adjclose'][0]['adjclose']
            i = 0
            prices_list = []
            for timestamp in timestamp_list:
                price_dict = {}
                price_dict.update({'date': timestamp})
                price_dict.update({'high': high_price_list[i]})
                price_dict.update({'low': low_price_list[i]})
                price_dict.update({'open': open_price_list[i]})
                price_dict.update({'close': close_price_list[i]})
                price_dict.update({'volume': volume_list[i]})
                price_dict.update({'adjclose': adj_close_list[i]})
                prices_list.append(price_dict)
                i += 1
            ret_obj.update({'prices': prices_list})
        return ret_obj

    @staticmethod
    def _get_url_path(hist_obj, up_ticker):
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        url_path = base_url + up_ticker + '?symbol=' + up_ticker + '&period1=' + str(hist_obj['start']) + '&period2=' + \
                   str(hist_obj['end']) + '&interval=' + hist_obj['interval']
        url_path += '&events=div|split|earn&lang=en-US&region=US'
        return url_path

    def _is_instance(self):
        sub_dict_list = []
        for sub_dict in v:
            sub_dict['formatted_date'] = self.format_date(sub_dict['date'])
            sub_dict_list.append(sub_dict)
        dict_ent = {k: sub_dict_list}

    def _load_yahoo_data(self, ticker, hist_oj):
        url = self._BASE_YAHOO_URL + self._encode_ticker(ticker) + '/history?period1=' + str(hist_oj['start']) + \
              '&period2=' + str(hist_oj['end']) + '&interval=' + hist_oj['interval'] + '&filter=history&frequency=' + \
              hist_oj['interval']
        return url

    def _process_load_stock_data(self, api_url, tries=0):
        urlopener = UrlOpener()
        response = urlopener.open(api_url)
        if response.getcode() == 200:
            res_content = response.read()
            response.close()
            if sys.version_info < (3, 0):
                return loads(res_content)
            return loads(res_content.decode('utf-8'))
        else:
            if tries < 6:
                time.sleep(random.randrange(5, 25))
                tries += 1
                return self._process_load_stock_data(api_url, tries)
            else:
                return None

    def _recursive_data_request(self, hist_obj, up_ticker, i=0):
        api_url = self._get_url_path(hist_obj, up_ticker)
        re_data = self._get_clean_data(api_url)
        cleaned_re_data = self._clean_stock_data(re_data)
        return cleaned_re_data

    @staticmethod
    def format_date(in_date):
        if isinstance(in_date, str):
            form_date = int(calendar.timegm(time.strptime(in_date, '%Y-%m-%d')))
        else:
            form_date = str((datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=in_date)).date())
        return form_date

    def get_past_data(self, start_date, end_date, time_interval):
        interval_code = self.get_time_code(time_interval)
        start = self.format_date(start_date)
        end = self.format_date(end_date)
        hist_obj = {'start': start, 'end': end, 'interval': interval_code}
        return self.get_stock_data('history', hist_obj=hist_obj)

    def get_stock_data(self, statement_type='income', report_name='', hist_obj={}):
        data = {}
        if isinstance(self.ticker, str):
            dict_ent = self._get_dict(self.ticker, statement_type, report_name, hist_obj)
            data.update(dict_ent)
        else:
            for tick in self.ticker:
                dict_ent = self._get_dict(tick, statement_type, report_name, hist_obj)
                data.update(dict_ent)
        return data

    def get_time_code(self, time_interval):
        interval_code = self._INTERVAL_DICT[time_interval.lower()]
        return interval_code
