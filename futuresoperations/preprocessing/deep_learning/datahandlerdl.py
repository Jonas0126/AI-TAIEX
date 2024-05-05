from futuresoperations.preprocessing.prototype import Prototype
from futuresoperations.preprocessing.deep_learning.datasplitter import DataSplitter
from futuresoperations.preprocessing.deep_learning.datascaler import Datascaler
from futuresoperations.preprocessing.deep_learning.financedataset import FinanceDataset
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

class DataHandlerDL(Prototype):
    def __init__(self, train_dates, test_dates,
                scaling, pca, correlation_value):
        super().__init__(train_dates, test_dates, scaling)
        self.pca = pca
        self.correlation_value = correlation_value
        self.datasplitter = DataSplitter(self.train_dates, self.test_dates)
        if correlation_value is not None:
            self.delete_col_dict = {}
        else:
            self.delete_col_dict = None
        if self.scaling:
            self.datascalers = {}
        else:
            self.datascalers = None

    def get_rawdata(self, target_name, selected_stocks):
        labels = pd.read_csv(f'./data/daily/{target_name}.csv')
        features = []
        for i in range(len(selected_stocks)):
            stock = pd.read_csv(f'./data/daily/{selected_stocks[i]}.csv')
            features.append(stock)
        return labels, features

    def load_data(self, target_name, selected_stocks):
        '''
        Get raw data and combine them into needed data.
        The dates of needed data is from train start to latest.
        '''
        labels, features = self.get_rawdata(target_name, selected_stocks)

        # range labels data from train start date to latest
        labels_volume = labels[['Date', 'Volume']]
        labels = labels[['Date', 'Open']]
        start_date = self.train_dates['start']
        last_date = labels['Date'][len(labels)-1]
        self.labels = labels[
            (labels['Date'] >= start_date)
            & (labels['Date'] <= last_date)
        ]

        # Add target volume to features
        self.features = labels_volume

        # combine raw data into needed data
        for i in range(len(features)):
            feature_name = selected_stocks[i]
            feature = features[i][['Date', 'Close']]
            feature = feature[
                (feature['Date'] >= start_date)
                & (feature['Date'] <= last_date)
            ]
            # add Date for debugging and easy understanding
            feature.columns = ['Date', feature_name]
            feature = feature.drop(columns='Date')
            self.features[feature_name] = feature
        
        return self.labels, self.features

    def create_dataset(self, data):
        features, labels = data
        dataset = FinanceDataset(features, labels)
        return dataset

    def get_correlation_data(self, labels, features, predict_period):
        delete_col_list = []
        train_feats, train_labels = self.datasplitter.range_train_data_for_correlation(
            labels=labels,
            features=features,
            start_date=self.train_dates['start'],
            end_date=self.train_dates['end'],
            predict_period=predict_period
        )
        for col in train_feats.columns:
            feat = train_feats[col].to_numpy()
            label = train_labels['Open'].to_numpy()
            correlation, p = spearmanr(feat, label)
            if correlation < self.correlation_value and correlation > -self.correlation_value:
                delete_col_list.append(col)
        features = features.drop(columns=delete_col_list)
        self.delete_col_dict[predict_period] = delete_col_list
        return features

    def data_scaling(self, features, predict_period):
        features_range = self.datasplitter.range_train_features_for_scaling(
            features=features,
            start_date=self.train_dates['start'],
            end_date=self.train_dates['end']
        )
        scaler = Datascaler()
        scaler.scaling(features_range, self.pca)
        temp_date = features['Date']
        features = features.drop(columns=['Date'])
        transformed_features = scaler.transform(features, self.pca)
        transformed_features = pd.DataFrame(transformed_features)
        transformed_features['Date'] = temp_date.to_list()
        self.datascalers[str(predict_period)] = scaler
        return transformed_features

    def data_preprocessing(self, labels, features,
                            predict_period, lookbacks=None):
        traindatasets = []
        testdatasets = []
        if self.correlation_value is not None:
            features = self.get_correlation_data(
                labels=labels,
                features=features,
                predict_period=predict_period
            )
        # DL data scaling first
        if self.scaling:
            features = self.data_scaling(features, predict_period)
            temp_date = labels['Date']
            labels = labels.drop(columns=['Date'])
            labels /= 10000
            labels['Date'] = temp_date.to_list()

        for lookback in lookbacks:
            train_data, test_data = self.datasplitter.train_test_split(
                labels=labels,
                features=features,
                predict_period=predict_period,
                lookback=lookback
            )
            traindataset = self.create_dataset(train_data)
            testdataset = self.create_dataset(test_data)
            traindatasets.append(traindataset)
            testdatasets.append(testdataset)

        return traindatasets, testdatasets
    
    def split_from_date(self, start_date, end_date,
                        predict_period, lookback=None):
        features = self.features
        labels = self.labels
        if self.correlation_value is not None:
            features = self.get_correlation_data(
                labels=labels,
                features=features,
                predict_period=predict_period
            )
        if self.scaling:
            features = self.data_scaling(features, predict_period)
            temp_date = labels['Date']
            labels = labels.drop(columns=['Date'])
            labels /= 10000
            labels['Date'] = temp_date.to_list()
        features, labels = self.datasplitter.range_test_data(
            labels1=labels,
            features1=features,
            start_date=start_date,
            end_date=end_date,
            lookback=lookback,
            predict_period=predict_period
        )
        data = (features, labels)
        return self.create_dataset(data)

    def split_for_inference(self, pred_sdate, pred_edate,
                            predict_period, lookback=None):
        data_x = []
        data_y = []
        needed_features = self.features.drop(columns='Date')
        needed_labels = self.labels.drop(columns='Date')
        if self.correlation_value is not None:
            delete_col = self.delete_col_dict[predict_period]
            needed_features = needed_features.drop(columns=delete_col)
        if self.scaling:
            scaler = self.datascalers[str(predict_period)]
            needed_features = scaler.transform(needed_features, self.pca)
            needed_labels /= 10000
        pred_dates = pd.date_range(
            start=pred_sdate,
            end=pred_edate,
            freq='MS'
        )
        pred_dates = pred_dates.strftime('%Y-%m-%d')
        for start_date in pred_dates:
            idx_list = self.features.index[
                self.features['Date'] >= start_date
            ].tolist()
            if len(idx_list) <= 0:
                idx = len(self.features)
            else:
                idx = idx_list[0]
            sindex = idx - lookback - (predict_period-1)*22
            eindex = sindex + lookback
            features = needed_features[sindex:eindex]
            labels = needed_labels[idx:idx+22]
            if len(labels) != 22:
                print("Use wrong labels!!!")
                labels = needed_labels[len(needed_labels)-22:len(needed_labels)]
            data_x.append(features)
            data_y.append(labels)
        data_x = np.array(data_x)
        data_y = np.array(data_y).squeeze()
        return self.create_dataset((data_x, data_y))

    def split_for_finetune(self, ):
        data_x = []
        data_y = []
        needed_features = self.features.drop(columns='Date')
        needed_labels = self.labels.drop(columns='Date')
        if self.correlation_value is not None:
            delete_col = self.delete_col_dict[predict_period]
            needed_features = needed_features.drop(columns=delete_col)
        if self.scaling:
            scaler = self.datascalers[str(predict_period)]
            needed_features = scaler.transform(needed_features, self.pca)
            needed_labels /= 10000
        

if __name__ == '__main__':
    dh = DataHandlerDL(
        train_dates="2020-01-01",
        test_dates="2020-12-31",
        scaling=True,
        pca=False,
        correlation_value=None
    )
    labels, features = dh.load_data(
        target_name='E-miniSP500',
        selected_stocks=['SP500']
    )
    print('labels:', labels)
    print('features:', features)
    