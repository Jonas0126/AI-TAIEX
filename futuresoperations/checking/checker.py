from futuresoperations.checking.exceptions import ModelNotFoundException, MLhyperparamsException, DLhyperparamsException
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

def check_model(model_type, hyperparams, scaling):
    models = ['LR', 'XGB', 'LSTM', 'TCN', 'CRNN']
    mlmodel = ['LR', 'XGB']
    dlmodel = ['LSTM', 'TCN', 'CRNN']
    try:
        if model_type not in models:
            raise ModelNotFoundException
        if model_type in mlmodel and 'lookbacks' in hyperparams:
            raise MLhyperparamsException
        if model_type in dlmodel and 'lookbacks' not in hyperparams:
            raise DLhyperparamsException
    except ModelNotFoundException:
        print(f'We do not support {model_type} model!\n')
        sys.exit(1)
    except MLhyperparamsException:
        print('We do not recommend that Machine Learning has lookbacks hyperparameter!\n')
        sys.exit(1)
    except DLhyperparamsException:
        print('Deep Learning Model should have lookbacks hyperparameter!\n')
        sys.exit(1)

def iscomplete_previous_data(stock_dict):
    print('Check previous data...')
    today_dt = datetime.now()
    this_month = f'{today_dt.strftime("%Y-%m")}-01'
    day_offset = 1
    if today_dt.isoweekday() > 5:
        day_offset = today_dt.isoweekday() - 4
    nearest_bs_last_date = today_dt - timedelta(days=day_offset)
    nearest_bs_last_date = nearest_bs_last_date.strftime("%Y-%m-%d")
    data_daily_path = './data/daily/'
    if os.path.exists(data_daily_path):
        dirs = os.listdir(data_daily_path)
        for stock_name in stock_dict.values():
            if f'{stock_name}.csv' not in dirs:
                print(f'{stock_name}.csv not in {dirs}')
                print('Data need to be updated.')
                return False
        daily_files = [f for f in dirs if os.path.isfile(os.path.join(data_daily_path, f))]
        for daily_f in daily_files:
            df = pd.read_csv(os.path.join(data_daily_path, daily_f))
            if len(df) == 0:
                print(f'{daily_f} no data')
                print('Data need to be updated.')
                return False
            if len(df) == 0 or df['Date'][len(df)-1] < '2022-12-30': # nearest_bs_last_date
                print(f'{daily_f} Date: {df["Date"][len(df)-1]} < nearest bs last date: {nearest_bs_last_date}')
                print('Data need to be updated.')
                return False
    else:
        print('Data need to be updated.')
        return False

    data_monthly_path = './data/monthly/'
    if os.path.exists(data_monthly_path):
        dirs = os.listdir(data_monthly_path)
        for stock_name in stock_dict.values():
            if f'{stock_name}.csv' not in dirs:
                print('Data need to be updated.')
                return False
        monthly_files = [f for f in dirs if os.path.isfile(os.path.join(data_monthly_path, f))]
        for monthly_f in monthly_files:
            df = pd.read_csv(os.path.join(data_monthly_path, monthly_f))
            if len(df) == 0:
                print(f'{monthly_f} no data')
                print('Data need to be updated.')
                return False
            if df['Date'][len(df)-1] < '2022-12':
                print('Data need to be updated.')
                return False
    else:
        print('Data need to be updated.')
        return False

    print('Use previous data...')
    return True