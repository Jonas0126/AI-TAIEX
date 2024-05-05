# import util.datachecker as datachecker
import numpy as np
import sys

class DataSplitter:
    def __init__(self, train_dates, test_dates):
        self.train_dates = train_dates
        self.test_dates = test_dates

    def range_train_data(self, labels1, features1, start_date,
                    end_date, lookback, predict_period=1):
        data_x = []
        data_y = []
        labels = labels1
        features = features1
        labels_range = labels[
            (labels['Date'] >= start_date)
            & (labels['Date'] <= end_date)
        ].drop(columns='Date')
        features_range = features[
            (features['Date'] >= start_date)
            & (features['Date'] <= end_date)
        ].drop(columns='Date')
        for idx in range(len(features_range)-lookback-1-predict_period*22):
            data_x.append(features_range[idx : idx+lookback])
            labels_start = idx + lookback + (predict_period-1)*22
            labels_end = labels_start + 22
            data_y.append(labels_range[labels_start:labels_end])
        data_x = np.array(data_x)
        data_y = np.array(data_y).squeeze()
        return data_x, data_y
    
    def range_test_data(self, labels1, features1, start_date,
                    end_date, lookback, predict_period=1):
        data_x = []
        data_y = []
        labels = labels1.drop(columns='Date')
        features = features1.drop(columns='Date')
        index_list = features1.index[
            (features1['Date'] >= start_date)
            & (features1['Date'] <= end_date)
        ].tolist()
        test_start_idx = index_list[0]
        test_end_idx = index_list[len(index_list)-1]
        '''
        test_start_idx-22: previous month data is not trained
        test_end_idx-22  : the last month data is not needed
        '''
        for idx in range(test_start_idx-22, test_end_idx-22):
            sindex = idx - lookback - (predict_period-1)*22
            eindex = sindex + lookback
            data_x.append(features[sindex:eindex])
            data_y.append(labels[idx:idx+22])
        data_x = np.array(data_x)
        data_y = np.array(data_y).squeeze()
        return data_x, data_y

    def range_train_data_for_correlation(self, labels, features,
                                        start_date, end_date, predict_period):
        labels_range = labels[
            (labels['Date'] >= start_date)
            & (labels['Date'] <= end_date)
        ].drop(columns='Date')
        features_range = features[
            (features['Date'] >= start_date)
            & (features['Date'] <= end_date)
        ].drop(columns='Date')
        features_range = features_range[:-predict_period*22]
        labels_range = labels_range[predict_period*22:]
        return features_range, labels_range
    
    def range_train_features_for_scaling(self, features, start_date, end_date):
        features_range = features[
            (features['Date'] >= start_date)
            & (features['Date'] <= end_date)
        ].drop(columns='Date')
        return features_range

    def train_test_split(self, labels, features, lookback, predict_period):
        train_x, train_y = self.range_train_data(
            labels1=labels,
            features1=features,
            start_date=self.train_dates['start'],
            end_date=self.train_dates['end'],
            lookback=lookback,
            predict_period=predict_period
        )
        
        test_x, test_y = self.range_test_data(
            labels1=labels,
            features1=features,
            start_date=self.test_dates['start'],
            end_date=self.test_dates['end'],
            lookback=lookback,
            predict_period=predict_period
        )

        train_data = (train_x, train_y)
        test_data = (test_x, test_y)
        return train_data, test_data