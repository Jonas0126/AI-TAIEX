from abc import ABC, abstractmethod

'''the time range of train and test data:
    Suppose DL
    train_dates = {
        'start': '2008-01-01',
        'end': '2020-12-31'
    }
    test_dates = {
        'start': '2021-01-01',
        'end': '2021-06-30'
    }
    Then if the prediction period is 1 month
    the time range of training features is 2008-01-01 ~ 2020-12-31-(period)*22-1
    the time range of training labels is   2008-01-01+lookback+(period-1)*22 ~ 2020-12-31
    so the time range of the first training pair:
        feature: 2008-01-01                        ~ 2008-01-01+lookback-1
        label:   2008-01-01+lookback+(period-1)*22 ~ 2008-01-01+lookback+(period-1)*22+22
    so the time range of the last training pair:
        feature: 2020-12-31-(period)*22-1-lookback ~ 2020-12-31-(period)*22-1
        label:   2020-12-31-(period)*22            ~ 2020-12-31
    the time range of testing features is  2021-01-01 ~ 2021-06-30-(period)*22
    the time range of testing labels is    2021-01-01+lookback ~ 2021-06-30

    Suppose ML
    train_dates = {
        'start': '2008-01-01',
        'end': '2020-12-31'
    }
    test_dates = {
        'start': '2021-01-01',
        'end': '2021-06-30'
    }
    (train_dates end and test_dates start must be consecutive)
    Then if the prediction period is 1 month
    the time range of training features is 2008-01-01 ~ 2020-11-30
    the time range of training labels is   2008-02-01 ~ 2020-12-31
    the time range of testing features is  2020-12-01 ~ 2021-05-30
    the time range of testing labels is    2021-01-01 ~ 2021-06-30
'''

class Prototype(ABC):
    '''abstract class
    '''
    def __init__(self, train_dates, test_dates, scaling):
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.scaling = scaling
        self.labels = None
        self.features = None
    
    @abstractmethod
    def load_data(self, target_name, selected_stocks):
        pass
    
    @abstractmethod
    def data_preprocessing(self, labels, features,
                            predict_period, lookbacks=None):
        pass

    # for finetune_when_trade
    @abstractmethod
    def split_from_date(self, start_date, end_date,
                        predict_period, lookback=None):
        pass

    @abstractmethod
    def split_for_inference(self, pred_sdate, pred_edate,
                            predict_period, lookback=None):
        pass
