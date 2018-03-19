import pandas as pd

from alphai_delphi import AbstractDataSource


class DataSource(AbstractDataSource):

    def __init__(self, configuration):
        raw_data_file = configuration['data_file']

        dataframe = pd.DataFrame()

        with pd.HDFStore(raw_data_file) as store:
            dataframe = store.get('data')

        dataframe.index = dataframe.index.map(lambda t: t.replace(hour=7))
        dataframe.index = dataframe.index.tz_localize('UTC')
        features = set(dataframe.columns) - {'DateStamps', 'Ticker'}
        self._data = {feature: dataframe.pivot(columns='Ticker', values=feature) for feature in features}

    def get_data(self, current_datetime, interval):

        start_date = current_datetime - (interval * 31)
        end_date = current_datetime

        data = {
            key: value[start_date:end_date] for key, value in self._data.items()
        }

        return data

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):

        try:
            feature_data = self._data[feature]
        except KeyError:
            raise KeyError("Feature {} doesn't exists in data".format(feature))

        symbol_data = feature_data[symbol_list]

        return symbol_data.loc[current_datetime]

    @property
    def start(self):
        pass

    @property
    def end(self):
        pass
