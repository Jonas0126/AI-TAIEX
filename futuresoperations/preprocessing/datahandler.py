from futuresoperations.preprocessing.prototype import Prototype
from futuresoperations.preprocessing.machine_learning import DataHandlerML
from futuresoperations.preprocessing.deep_learning import DataHandlerDL

class DataHandler(Prototype):
    def __init__(self, train_dates, test_dates, scaling,
                model_type, pca, correlation_value=None):
        super().__init__(train_dates, test_dates, scaling)
        self.model_type = model_type
        self.pca = pca
        self.correlation_value = correlation_value
        self.handler = self.select_handler()
    
    def select_handler(self):
        ml_model = ['LR', 'XGB']
        if self.model_type in ml_model:
            return DataHandlerML(
                train_dates=self.train_dates,
                test_dates=self.test_dates,
                scaling=self.scaling,
                pca=self.pca,
                correlation_value=self.correlation_value
            )
        else:
            return DataHandlerDL(
                train_dates=self.train_dates,
                test_dates=self.test_dates,
                scaling=self.scaling,
                pca=self.pca,
                correlation_value=self.correlation_value
            )
    
    def load_data(self, target_name, selected_stocks):
        labels, features = self.handler.load_data(
            target_name=target_name,
            selected_stocks=selected_stocks
        )
        self.labels = labels
        self.features = features
        return self.labels, self.features

    def get_correlation_stocks(self, stocks, y_start, y_end, threshold=0.5):
        # TODO
        return

    def data_preprocessing(self, labels, features,
                            predict_period, lookbacks=None):
        traindatasets, testdatasets = self.handler.data_preprocessing(
            labels=labels,
            features=features,
            predict_period=predict_period,
            lookbacks=lookbacks
        )
        return traindatasets, testdatasets

    def split_from_date(self, start_date, end_date,
                        predict_period, lookback=None):
        return self.handler.split_from_date(
            start_date=start_date,
            end_date=end_date,
            predict_period=predict_period,
            lookback=lookback
        )
    
    def split_for_inference(self, pred_sdate, pred_edate,
                            predict_period, lookback=None):
        return self.handler.split_for_inference(
            pred_sdate=pred_sdate,
            pred_edate=pred_edate,
            predict_period=predict_period,
            lookback=lookback
        )


if __name__ == '__main__':
    train_dates = {
        'start': '2008-01-01',
        'end': '2020-12-31'
    }
    test_dates = {
        'start': '2021-01-01',
        'end': '2021-06-30'
    }
    target_name='Taiex'
    selected_stocks=['AORD', 'Taiex']
    dh = DataHandler(train_dates=train_dates, test_dates=test_dates)
    dh.load_data(target_name, selected_stocks)
    print(dh.labels)
    print(dh.features)
    x_train, y_train, x_test, y_test = dh.train_test_split(1)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)