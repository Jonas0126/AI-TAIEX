import os
import sys
import argparse
import pandas as pd
import yaml
from utils.check import check_config_file
from futuresoperations import FuturesOperator
from tradingsimulation import TradingSimulator

class Ensembler():
    def __init__(self, fp1, fp2, train_dates, test_dates,
                target_path, scaling, correlation=None, predict_range=3):
        self.fp1 = fp1
        self.fp2 = fp2
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.model_type = fp1.model_type + '+' + fp2.model_type
        self.model_path = target_path
        self.correlation = correlation
        self.scaling = scaling
        self.predict_range = predict_range
        self.__create_model_path()

    def __create_model_path(self):
        train_time_range = 'train_'
        train_time_range += f'{self.train_dates["start"]}_{self.train_dates["end"]}'
        test_time_range = 'test_'
        test_time_range += f'{self.test_dates["start"]}_{self.test_dates["end"]}'
        folder_name = f'{train_time_range}_{test_time_range}'
        self.model_path = os.path.join(self.model_path, self.model_type)
        self.model_path = os.path.join(self.model_path, folder_name)
        if self.correlation is not None:
            self.model_path = os.path.join(self.model_path, f'correlation{self.correlation}')
        if self.scaling:
            self.model_path = os.path.join(self.model_path, 'scaling')
        os.makedirs(self.model_path, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='give a config file for operations')
    parser.add_argument('-t', '--target', nargs='?', type=str, default='Taiex', help='choose target for analysis')

    args = parser.parse_args()
    config_data = None
    with open(args.config, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    target = args.target
    target_path = './models/target_' + target + '/'
    train_dates = config_data['train dates']
    test_dates = config_data['test dates']
    predict_range = config_data['predict range']
    features = config_data['features']
    correlation = 0.5
    scaling = True

    if os.path.exists(target_path):
        for ML_model in config_data['ML models']:
            ML_model_path = os.path.join(target_path, ML_model)
            for DL_model in config_data['DL models']:
                DL_model_path = os.path.join(target_path, DL_model)
                for correlation in config_data['preprocessing']['correlations']:
                    for scaling in config_data['preprocessing']['scaling']:
                        DL_fp = FuturesOperator(
                            train_dates=train_dates,
                            test_dates=test_dates,
                            target_name=target,
                            stock_dict=features,
                            correlation=correlation,
                            scaling=scaling,
                            predict_range=predict_range,
                            model_type=DL_model,
                            hyperparams=config_data['DL hyperparameters']
                        )
                        ML_fp = FuturesOperator(
                            train_dates=train_dates,
                            test_dates=test_dates,
                            target_name=target,
                            stock_dict=features,
                            correlation=correlation,
                            scaling=scaling,
                            predict_range=predict_range,
                            model_type=ML_model,
                            hyperparams=config_data['ML hyperparameters']
                        )
                        eb = Ensembler(
                            fp1=ML_fp,
                            fp2=DL_fp,
                            train_dates=train_dates,
                            test_dates=test_dates,
                            target_path=target_path,
                            scaling=scaling,
                            correlation=correlation,
                            predict_range=predict_range
                        )
                        ts = TradingSimulator(
                            start_date=config_data['trade dates']['start'],
                            end_date=config_data['trade dates']['end'],
                            target_name=config_data['target name'],
                            contract_name=config_data['futures name'],
                            futures_rule=config_data['futures rule'],
                            emsembler=eb
                        )
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
    else:
        print(f'{target_path} not found!!!')
        sys.exit(1)