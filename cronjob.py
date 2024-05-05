"""
The job is for updating the latest prediction and is executed on the last day of every month
"""

import sys
import yaml
import json
import copy
import pickle
import argparse
import pandas as pd
from datetime import datetime, timedelta
from futuresoperations import FuturesOperator
from futuresoperations.preparing import DataCrawler
from futuresoperations.preprocessing import DataHandler

# TODAY = datetime.today()
TODAY = datetime(2023, 11, 30)

def is_the_last_day_of_this_month():
    next_month = TODAY.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    return TODAY.year == last_day.year and TODAY.month == last_day.month and TODAY.day == last_day.day

def need_retrain():
    if TODAY.day == 31:
        if TODAY.month == 12:
            return True
        else:
            return False
    else:
        return False

def do_retrain(config):
    print('do retrain')
    train_dates = {
        'start': '2008-01-01',
        'end': f'{TODAY.year-1}-12-31'
    }
    print('train dates:', train_dates)
    test_dates = {
        'start': f'{TODAY.year}-01-01',
        'end': TODAY.strftime('%Y-%m-%d')
    }
    print('test dates:', test_dates)
    fp = FuturesOperator(
        train_dates=train_dates,
        test_dates=test_dates,
        trade_dates=None,
        target_name=config['target name'],
        stock_dict=config['features'],
        correlation=None,
        scaling=True,
        pca=True,
        predict_range=3,
        model_type='CRNN',
        hyperparams=config['hyperparameters']
    )
    fp.run()
    # copy model for fintuning in the future
    for predict_period in range(1, fp.predict_range+1):
        model_name = 'CRNN_'
        model_name += f'predmonth_{predict_period}'
        model = pickle.load(open(f'{fp.model_path}/{model_name}.pkl', 'rb'))
        pickle.dump(model, open(f'{fp.model_path}/{model_name}_latest.pkl', 'wb'))

    # save next month prediction
    pred_date = f'{TODAY.year+1}-01'
    pred_df = fp.inference_by_load(pred_date, pred_date)

    # pred_date = (TODAY+timedelta(days=34)).strftime('%Y-%m') # next second month
    # pred_df2 = add_other_month_prediction(pred_date, pred_date, 2, fp)
    # pred_df = pd.concat([pred_df, pred_df2], ignore_index=True)

    # pred_date = (TODAY+timedelta(days=65)).strftime('%Y-%m') # next third month
    # pred_df3 = add_other_month_prediction(pred_date, pred_date, 3, fp)
    # pred_df = pd.concat([pred_df, pred_df3], ignore_index=True)

    pred_df.to_csv(f'{fp.model_path}/prediction.csv', index=False)
    print(pred_df)

def do_finetune(config):
    print('do finetune')
    train_dates = {
        'start': '2008-01-01',
        'end': f'{TODAY.year-2}-12-31'
    }
    print('train dates:', train_dates)
    test_dates = {
        'start': f'{TODAY.year-1}-01-01',
        'end': f'{TODAY.year-1}-12-31'
    }
    print('test dates:', test_dates)
    fp = FuturesOperator(
        train_dates=train_dates,
        test_dates=test_dates,
        trade_dates=None,
        target_name=config['target name'],
        stock_dict=config['features'],
        correlation=None,
        scaling=True,
        pca=True,
        predict_range=3,
        model_type='CRNN',
        hyperparams=config['hyperparameters']
    )
    # 1. need crawl data
    datacrawler = DataCrawler(
        start_date=train_dates['start'],
        fill_missing=True
    )
    datacrawler.crawl_data(config['features'])
    dh = DataHandler(
        train_dates=train_dates,
        test_dates=test_dates,
        scaling=True,
        model_type='CRNN',
        pca=True,
        correlation_value=None
    )
    dh.load_data(
        target_name=config['target name'],
        selected_stocks=list(config['features'].values())
    )
    for predict_period in range(1, fp.predict_range+1):
        model_name = 'CRNN_'
        model_name += f'predmonth_{predict_period}'
        model = pickle.load(open(f'{fp.model_path}/{model_name}_latest.pkl', 'rb'))
        lookback = None
        if 'lookback' in model.best_hyperparams:
            lookback = model.best_hyperparams['lookback']
        finetunedataset = dh.split_from_date(
            start_date=f'{TODAY.strftime("%Y-%m")}-01',
            end_date=TODAY.strftime('%Y-%m-%d'),
            predict_period=predict_period,
            lookback=lookback
        )
        model.finetuning(finetunedataset)
        pickle.dump(model, open(f'{fp.model_path}/{model_name}_latest.pkl', 'wb'))
        fp.models.append([copy.deepcopy(model)])

    # save next month prediction
    pred_date = (TODAY+timedelta(days=3)).strftime('%Y-%m') # next month
    pred_df = fp.inference_by_load(pred_date, pred_date, 3)
    df = pd.read_csv(f'{fp.model_path}/prediction.csv')
    print(f'prediction.csv : \n{df}')
    pred_df = pd.concat([df, pred_df], ignore_index=True)
    print(f'next month : \n{pred_df}')
    pred_date = (TODAY+timedelta(days=34)).strftime('%Y-%m') # next second month
    pred_df2 = add_other_month_prediction(pred_date, pred_date, 2, fp)
    pred_df = pd.concat([pred_df, pred_df2], ignore_index=True)
    print(f'next second month : \n{pred_df}')
    pred_date = (TODAY+timedelta(days=65)).strftime('%Y-%m') # next third month
    pred_df3 = add_other_month_prediction(pred_date, pred_date, 3, fp)
    pred_df = pd.concat([pred_df, pred_df3], ignore_index=True)
    pred_df.to_csv(f'{fp.model_path}/prediction.csv', index=False)
    print(f'next third month : \n{pred_df}')

def add_other_month_prediction(sdate, edate, predict_period, fp):
    """
    for Taiex
    """
    model = fp.load_model(predict_period)
    fp.load_dh(predict_period)
    lookback = None
    if 'lookback' in model.best_hyperparams:
        lookback = model.best_hyperparams['lookback']
    prediction = []
    rmse = 0
    pred_df = pd.DataFrame(columns=['Features Date', 'Predict Date', 'Predict'])
    trade_dates_month_begin = pd.date_range(
        start=sdate,
        end=edate,
        freq='MS'
    )
    accurate_trade_end = trade_dates_month_begin[len(trade_dates_month_begin)-1]
    accurate_trade_end = accurate_trade_end + pd.DateOffset(months=1)
    accurate_trade_end = accurate_trade_end.strftime('%Y-%m')
    trade_dates_month_end = pd.date_range(
        start=sdate,
        end=accurate_trade_end,
        freq='M'
    )
    trade_dates_month_begin = trade_dates_month_begin.strftime('%Y-%m-%d')
    trade_dates_month_end = trade_dates_month_end.strftime('%Y-%m-%d')
    for idx in range(len(trade_dates_month_begin)):
        dataset = fp.dh.split_for_inference(
            pred_sdate=trade_dates_month_begin[idx],
            pred_edate=trade_dates_month_end[idx],
            predict_period=predict_period,
            lookback=lookback
        )
        pred, r = fp.models[predict_period-1][idx].inference(dataset)
        prediction += pred
        rmse += r
    rmse /= len(trade_dates_month_begin)
    prediction = [pred*10000 for pred in prediction]
    predict_date = pd.date_range(
        start=sdate,
        end=edate,
        freq='MS'
    )
    features_date = predict_date - pd.DateOffset(months=predict_period)
    predict_date = predict_date.strftime('%Y-%m').to_list()
    features_date = features_date.strftime('%Y-%m').to_list()
    df = {
        'Features Date': features_date,
        'Predict Date': predict_date,
        'Predict': prediction
    }
    df = pd.DataFrame(data=df)
    print('Predict period:', predict_period)
    print('Predict:', prediction)
    print('RMSE:', rmse)
    pred_df = pd.concat([pred_df, df], ignore_index=True)
    return pred_df

def do_simulate():
    '''
    
    '''
    print('do simulate')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='give a config file for operations')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if not is_the_last_day_of_this_month():
        print("Crontab wrong!!!")
        sys.exit(1)

    if need_retrain():
        do_retrain(config)
    else:
        do_finetune(config)