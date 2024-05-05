from futuresoperations.preprocessing.machine_learning.util import datacalendar, datachecker
import numpy as np

class DataSplitter:
    def __init__(self, train_dates, test_dates):
        self.train_dates = train_dates
        self.test_dates = test_dates
    
    def range_data(self, df, start_date, end_date,
                   predict_period=1, size_type='all'):
        '''Range the data and check the missing value
        '''
        data = df[
            (df['Date'] >= start_date)
            & (df['Date'] <= end_date)
        ]
        ismatch = datachecker.check_data_size(
            df=data,
            train_dates=self.train_dates,
            test_dates=self.test_dates,
            start_date=start_date,
            end_date=end_date,
            predict_period=predict_period,
            size_type=size_type
        )
        if ismatch:
            return data
        else:
            return None

    def get_train_feats(self, features, predict_period):
        feat_train_dates, feat_test_dates = datacalendar.calculate_dates(
            train_dates=self.train_dates,
            test_dates=self.test_dates,
            data_type='feature',
            predict_period=predict_period
        )
        train_stocks = self.range_data(
            df=features,
            start_date=feat_train_dates['start'],
            end_date=feat_train_dates['end'],
            predict_period=predict_period,
            size_type='train'
        )
        return train_stocks

    def get_train_labels(self, labels, predict_period):
        label_train_dates, label_test_dates = datacalendar.calculate_dates(
            train_dates=self.train_dates,
            test_dates=self.test_dates,
            data_type='label',
            predict_period=predict_period
        )
        train_labels = self.range_data(
            df=labels,
            start_date=label_train_dates['start'],
            end_date=label_train_dates['end'],
            predict_period=predict_period,
            size_type='train'
        )
        return train_labels

    def train_test_split(self, labels, features, predict_period, scaling):
        label_train_dates, label_test_dates = datacalendar.calculate_dates(
            train_dates=self.train_dates,
            test_dates=self.test_dates,
            data_type='label',
            predict_period=predict_period
        )
        feat_train_dates, feat_test_dates = datacalendar.calculate_dates(
            train_dates=self.train_dates,
            test_dates=self.test_dates,
            data_type='feature',
            predict_period=predict_period
        )
        
        train_stocks = self.range_data(
            df=features,
            start_date=feat_train_dates['start'],
            end_date=feat_train_dates['end'],
            predict_period=predict_period,
            size_type='train'
        )
        
        test_stocks = self.range_data(
            df=features,
            start_date=feat_test_dates['start'],
            end_date=feat_test_dates['end'],
            predict_period=predict_period,
            size_type='test'
        )
        
        train_labels = self.range_data(
            df=labels,
            start_date=label_train_dates['start'],
            end_date=label_train_dates['end'],
            predict_period=predict_period,
            size_type='train'
        )
        test_labels = self.range_data(
            df=labels,
            start_date=label_test_dates['start'],
            end_date=label_test_dates['end'],
            predict_period=predict_period,
            size_type='test'
        )

        x_train = train_stocks.drop(columns='Date').to_numpy()
        x_test = test_stocks.drop(columns='Date').to_numpy()
        if scaling:
            y_train = train_labels['Open'].to_numpy()/10000
            y_test = test_labels['Open'].to_numpy()/10000
        else:
            y_train = train_labels['Open'].to_numpy()
            y_test = test_labels['Open'].to_numpy()
        train_data = (x_train, y_train)
        test_data = (x_test, y_test)

        return train_data, test_data