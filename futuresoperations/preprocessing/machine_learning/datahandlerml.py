from futuresoperations.preprocessing.prototype import Prototype
from futuresoperations.preprocessing.machine_learning.datasplitter import DataSplitter
from futuresoperations.preprocessing.machine_learning.datascaler import Datascaler
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from scipy.stats import spearmanr

class DataHandlerML(Prototype):
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
        labels = pd.read_csv(f'./data/monthly/{target_name}.csv')
        features = []
        for i in range(len(selected_stocks)):
            stock = pd.read_csv(f'./data/monthly/{selected_stocks[i]}.csv')
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
        last_date = datetime.strptime(last_date, "%Y-%m-%d")
        last_date = str(date(last_date.year, last_date.month, 15))
        self.labels = self.datasplitter.range_data(
            df=labels,
            start_date=start_date,
            end_date=last_date,
            size_type='train2latest'
        )

        # Add target volume to features
        self.features = labels_volume

        # combine raw data into needed data
        for i in range(len(features)):
            feature_name = selected_stocks[i]
            feature = features[i][['Date', 'Close']]
            feature = self.datasplitter.range_data(
                df=feature,
                start_date=start_date,
                end_date=last_date,
                size_type='train2latest'
            )
            # add Date for debugging and easy understanding
            feature.columns = ['Date', feature_name]
            feature = feature.drop(columns='Date')
            self.features[feature_name] = feature
        
        return self.labels, self.features

    def get_correlation_data(self, labels, features, predict_period):
        delete_col_list = []
        train_feats = self.datasplitter.get_train_feats(features, predict_period)
        train_labels = self.datasplitter.get_train_labels(labels, predict_period)
        for col in train_feats.columns:
            feat = train_feats[col].to_numpy()
            label = train_labels['Open'].to_numpy()
            correlation, p = spearmanr(feat, label)
            if correlation < self.correlation_value and correlation > -self.correlation_value:
                delete_col_list.append(col)
        features = features.drop(columns=delete_col_list)
        self.delete_col_dict[predict_period] = delete_col_list
        return features

    def data_preprocessing(self, labels, features,
                            predict_period, lookbacks=None):
        traindatasets = []
        testdatasets = []
        new_features = features
        if self.correlation_value is not None:
            new_features = self.get_correlation_data(labels, features, predict_period)
        train_data, test_data = self.datasplitter.train_test_split(
            labels=labels,
            features=new_features,
            predict_period=predict_period,
            scaling=self.scaling
        )

        # data scaling
        if self.scaling:
            scaler = Datascaler()
            train_data, test_data = scaler.scaling(
                numpy_traindata=train_data,
                numpy_testdata=test_data,
                dopca=self.pca
            )
            self.datascalers[str(predict_period)] = scaler

        traindatasets.append(train_data)
        testdatasets.append(test_data)

        return traindatasets, testdatasets
    
    def split_from_date(self, start_date, end_date,
                        predict_period, lookback=None):
        features = self.datasplitter.range_data(
            df=self.features,
            start_date=start_date,
            end_date=end_date,
            predict_period=predict_period,
        )
        labels = self.datasplitter.range_data(
            df=self.labels,
            start_date=start_date,
            end_date=end_date,
            predict_period=predict_period,
        )
        return (features, labels)
    
    def split_for_inference(self, pred_sdate, pred_edate,
                            predict_period, lookback=None):
        '''
        labels date is the same as the predict date
        features date will be calculated by predict_period
        for example:
            predict date:  2021/1                  ~ 2021/6
            labels date:   2021/1                  ~ 2021/6
            features date: 2021/1 - predict_period ~ 2021/6 - predict_period
        '''
        sdate = datetime.strptime(pred_sdate, '%Y-%m')
        edate = datetime.strptime(pred_edate, '%Y-%m')
        edate = datetime(edate.year, edate.month, 14)
        feat_sdate = sdate - relativedelta(months=predict_period)
        feat_edate = edate - relativedelta(months=predict_period)
        feat_sdate = str(feat_sdate.date())
        feat_edate = str(feat_edate.date())
        label_sdate = str(sdate.date())
        label_edate = str(edate.date())
        features = self.features[
            (self.features['Date'] >= feat_sdate)
            & (self.features['Date'] <= feat_edate)
        ].drop(columns='Date')

        if self.delete_col_dict is not None:
            features = features.drop(columns=self.delete_col_dict[predict_period])
        features = features.to_numpy()
        if self.scaling:
            features = self.datascalers[str(predict_period)].transform(features, self.pca)
            labels = self.labels[
                (self.labels['Date'] >= label_sdate)
                & (self.labels['Date'] <= label_edate)
            ].drop(columns='Date').to_numpy()/10000
            return (features, labels)
        else:
            labels = self.labels[
                (self.labels['Date'] >= label_sdate)
                & (self.labels['Date'] <= label_edate)
            ].drop(columns='Date').to_numpy()
            return (features, labels)