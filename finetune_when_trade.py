from futuresoperations import FuturesOperator
from tradingsimulation import TradingSimulator
from utils.check import check_config_file
import pandas as pd
import argparse
import yaml
import copy
import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='give a config file for operations')
    args = parser.parse_args()

    config_data = None
    with open(args.config, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    check_config_file(config_data)

    # TODO: retrain when trading?
    if config_data['model'] == 'LR' or config_data['model'] == 'XGB':
        print('We do not support finetuning ML models!!!')
        sys.exit(1)

    for correlation in config_data['preprocessing']['correlations']:
        for scaling in config_data['preprocessing']['scaling']:
            for pca in config_data['preprocessing']['pca']:
                '''
                finetune type:
                    0. no finetune
                    1. finetune only in training
                    2. finetune in training and finetune once when trading
                    3. finetune in training and finetune constantly when trading
                '''
                finetune_types = [0, 2, 3]
                # finetune_types = [3]
                fp = FuturesOperator(
                    train_dates=config_data['train dates'],
                    test_dates=config_data['test dates'],
                    trade_dates=config_data['trade dates'],
                    target_name=config_data['target name'],
                    stock_dict=config_data['features'],
                    correlation=correlation,
                    scaling=scaling,
                    pca=pca,
                    predict_range=config_data['predict range'],
                    model_type=config_data['model'],
                    hyperparams=config_data['hyperparameters']
                )
                pred_df = fp.inference_by_load(
                    sdate=config_data['trade dates']['start'],
                    edate=config_data['trade dates']['end'],
                )
                pred_df.to_csv(os.path.join(fp.model_path, 'pred_df.csv'), index=False)
                for finetune_type in finetune_types:
                    fp = FuturesOperator(
                        train_dates=config_data['train dates'],
                        test_dates=config_data['test dates'],
                        trade_dates=config_data['trade dates'],
                        target_name=config_data['target name'],
                        stock_dict=config_data['features'],
                        correlation=correlation,
                        scaling=scaling,
                        pca=pca,
                        predict_range=config_data['predict range'],
                        model_type=config_data['model'],
                        hyperparams=config_data['hyperparameters']
                    )
                    if finetune_type >= 2:
                        for predict_period in range(1, fp.predict_range+1):
                            fp.load_dh(predict_period)
                            model = fp.load_model(predict_period)
                            lookback = None
                            if 'lookback' in model.best_hyperparams:
                                lookback = model.best_hyperparams['lookback']
                            # here we assume trading duration is 1 year
                            trade_dates_month_begin = pd.date_range(
                                start=config_data['trade dates']['start'],
                                end=config_data['trade dates']['end'],
                                freq='MS'
                            )
                            accurate_trade_end = trade_dates_month_begin[len(trade_dates_month_begin)-1]
                            accurate_trade_end = accurate_trade_end + pd.DateOffset(months=1)
                            accurate_trade_end = accurate_trade_end.strftime('%Y-%m')
                            trade_dates_month_end = pd.date_range(
                                start=config_data['trade dates']['start'],
                                end=accurate_trade_end,
                                freq='M'
                            )
                            trade_dates_month_begin = trade_dates_month_begin.strftime('%Y-%m-%d')
                            trade_dates_month_end = trade_dates_month_end.strftime('%Y-%m-%d')

                            models = []
                            # finetune once
                            if finetune_type == 2:
                                models.append(copy.deepcopy(model))
                                for idx in range(len(trade_dates_month_begin)-1):
                                    finetunedataset = fp.dh.split_from_date(
                                        start_date=trade_dates_month_begin[0],
                                        end_date=trade_dates_month_end[idx],
                                        predict_period=predict_period,
                                        lookback=lookback
                                    )
                                    m = fp.load_model(predict_period) # use load to prevent copy problem
                                    m.finetuning(finetunedataset)
                                    models.append(m)
                                fp.models.append(models)
                            else: # finetune constantly
                                models.append(copy.deepcopy(model))
                                for idx in range(len(trade_dates_month_begin)-1):
                                    finetunedataset = fp.dh.split_from_date(
                                        start_date=trade_dates_month_begin[idx],
                                        end_date=trade_dates_month_end[idx],
                                        predict_period=predict_period,
                                        lookback=lookback
                                    )
                                    model.finetuning(finetunedataset)
                                    models.append(copy.deepcopy(model))
                                fp.models.append(models)

                    ts = TradingSimulator(
                        start_date=config_data['trade dates']['start'],
                        end_date=config_data['trade dates']['end'],
                        target_name=config_data['target name'],
                        contract_name=config_data['futures name'],
                        futures_rule=config_data['futures rule'],
                        fp=fp,
                        finetune_type=finetune_type
                    )
                    pred_df = fp.inference_by_load(
                        sdate=config_data['trade dates']['start'],
                        edate=config_data['trade dates']['end'],
                        finetune_type=finetune_type
                    )
                    pred_df.to_csv(os.path.join(ts.trading_path, 'pred_df.csv'), index=False)
                    for trade_type in config_data['trading']['types']:
                        for stop_loss in config_data['trading']['stop loss']:
                            if stop_loss:
                                for tolerance in config_data['trading']['tolerance']:
                                    contracts = ts.simulation(
                                        trading_type=trade_type,
                                        stop_loss=stop_loss,
                                        tolerance=tolerance
                                    )
                                    print(contracts)
                            else:
                                contracts = ts.simulation(
                                    trading_type=trade_type,
                                    stop_loss=stop_loss,
                                    tolerance=0
                                )
                                print(contracts)