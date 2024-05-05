from futuresoperations.preprocessing.machine_learning.util import datacalendar as calendar

def get_check_date(train_dates, test_dates,
                    start_date, end_date, size_type):
    if size_type == 'train&test':
        return train_dates['start'], test_dates['end']
    elif size_type == 'train':
        return train_dates['start'], train_dates['end']
    elif size_type == 'test':
        return test_dates['start'], test_dates['end']
    else:
        return start_date, end_date
def check_data_size(df, train_dates, test_dates, start_date,
                            end_date, predict_period=1, size_type='all'):
    '''Check the missing value
    '''
    check_start_date, check_end_date = get_check_date(
        train_dates=train_dates,
        test_dates=test_dates,
        start_date=start_date,
        end_date=end_date,
        size_type=size_type
    )
    month_length = calendar.calculate_month_length(
        check_start_date,
        check_end_date
    )
    if size_type=='train&test' or size_type=='train':
        # for all and train data range, month_length should minus predict_period
        month_length -= predict_period
    
    # TODO: check missing value for exception
    if len(df) != month_length:
        print('The target monthly data have missing value!!!')
        print(df)
        print(month_length)
        return False
    else:
        return True

# def check_daily_data_size():