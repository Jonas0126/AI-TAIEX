from datetime import date, datetime
from dateutil.relativedelta import relativedelta
        
def calculate_dates(train_dates, test_dates, data_type, predict_period):
    '''Calculate the sdate and edate of train and test data
    '''
    sdate_train = datetime.strptime(train_dates['start'], "%Y-%m-%d")
    edate_train = datetime.strptime(train_dates['end'], "%Y-%m-%d")
    sdate_test = datetime.strptime(test_dates['start'], "%Y-%m-%d")
    edate_test = datetime.strptime(test_dates['end'], "%Y-%m-%d")
    if data_type == 'feature':
        sdate_train_feat = train_dates['start']
        edate_train_feat = edate_train - relativedelta(months=predict_period)
        edate_train_feat = str(edate_train_feat.date())

        sdate_test_feat = sdate_test - relativedelta(months=predict_period)
        sdate_test_feat = str(sdate_test_feat.date())
        edate_test_feat = edate_test - relativedelta(months=predict_period)
        edate_test_feat = str(edate_test_feat.date())

        train_dates_feat = {
            'start': sdate_train_feat,
            'end': edate_train_feat
        }
        test_dates_feat = {
            'start': sdate_test_feat,
            'end': edate_test_feat
        }
        return train_dates_feat, test_dates_feat
    else:
        sdate_train_label = sdate_train + relativedelta(months=predict_period)
        sdate_train_label = str(sdate_train_label.date())
        edate_train_label = train_dates['end']

        sdate_test_label = test_dates['start']
        edate_test_label = test_dates['end']

        train_dates_label = {
            'start': sdate_train_label,
            'end': edate_train_label
        }
        test_dates_label = {
            'start': sdate_test_label,
            'end': edate_test_label
        }
        return train_dates_label, test_dates_label

def calculate_month_length(start_date, end_date):
    '''Calculate the # of month between sdate and edate
    '''
    sdate = datetime.strptime(start_date, "%Y-%m-%d")
    edate = datetime.strptime(end_date, "%Y-%m-%d")
    month_length = 12 - sdate.month + 1
    month_length += edate.month
    month_length += (edate.year-sdate.year-1) * 12
    return month_length