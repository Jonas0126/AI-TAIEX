from futuresoperations import FuturesOperator
from tradingsimulation import TradingSimulator
from utils.check import check_config_file
import argparse
import yaml

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

    hyperparams = {
        # 'lookbacks': [132, 264, 396, 528, 660, 792]
    }

    for correlation in config_data['preprocessing']['correlations']:
        for scaling in config_data['preprocessing']['scaling']:
            for pca in config_data['preprocessing']['pca']:
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
                fp.run()

                ts = TradingSimulator(
                    start_date=config_data['trade dates']['start'],
                    end_date=config_data['trade dates']['end'],
                    target_name=config_data['target name'],
                    contract_name=config_data['futures name'],
                    futures_rule=config_data['futures rule'],
                    fp=fp
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
                        else:
                            contracts = ts.simulation(
                                trading_type=trade_type,
                                stop_loss=stop_loss,
                                tolerance=0
                            )