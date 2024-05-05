from futuresoperations.preparing import DataCrawler
from futuresoperations.preprocessing import DataHandler
from futuresoperations.training import FuturesModel
from futuresoperations.checking import checker
import pandas as pd
from tqdm import tqdm
import os
import pickle
import json

class FuturesOperator:
    def __init__(self, train_dates, test_dates, trade_dates, target_name, stock_dict, scaling,
                 pca, correlation=None, predict_range=3, model_type='LR', hyperparams={}):
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.trade_dates = trade_dates # for finetuning when trading (currently not used)
        self.target_name = target_name
        self.stock_dict = stock_dict
        self.scaling = scaling
        self.pca = pca
        self.predict_range = predict_range
        self.model_type = model_type
        if hyperparams is None:
            self.hyperparams = {}
        else:
            self.hyperparams = hyperparams
        self.correlation = correlation
        self.datacrawler = None
        self.models = []
        self.dh = None
        self.model_path = ''
        checker.check_model(self.model_type, self.hyperparams, self.scaling)
        self.__create_model_path()
    
    def __create_model_path(self):
        self.model_path = f'./models/target_{self.target_name}/{self.model_type}/'
        train_time_range = 'train_'
        train_time_range += f'{self.train_dates["start"]}_{self.train_dates["end"]}'
        test_time_range = 'test_'
        test_time_range += f'{self.test_dates["start"]}_{self.test_dates["end"]}'
        folder_name = f'{train_time_range}_{test_time_range}'
        self.model_path += f'{folder_name}'
        if self.correlation is not None:
            self.model_path += f'/correlation{self.correlation}'
        if self.scaling:
            self.model_path += '/scaling'
        if self.pca:
            self.model_path += '/pca'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def data_crawling(self):
        if not checker.iscomplete_previous_data(self.stock_dict):
            print('Start crawling data...\n')
            self.datacrawler = DataCrawler(
                start_date=self.train_dates['start'],
                fill_missing=True
            )
            self.datacrawler.crawl_data(self.stock_dict)
            print('Crawling data completes!\n')

    def save_dh(self, predict_period):
        model_name = f'{self.model_type}_'
        model_name += f'predmonth_{predict_period}'
        pickle.dump(self.dh, open(f'{self.model_path}/{model_name}_datahandler.pkl', 'wb'))
        if self.correlation is not None:
            json.dump(
                self.dh.handler.delete_col_dict,
                open(f'{self.model_path}/{model_name}_nouse_col.txt', 'w')
            )

    def load_dh(self, predict_period):
        model_name = f'{self.model_type}_'
        model_name += f'predmonth_{predict_period}'
        self.dh = pickle.load(open(f'{self.model_path}/{model_name}_datahandler.pkl', 'rb'))

    def save_model(self, model, predict_period, finetune=True):
        if finetune is True:
            model.draw_loss(self.model_path)
        model_name = f'{self.model_type}_'
        model_name += f'predmonth_{predict_period}'
        if finetune is False:
            model_name += '_no_finetune'
        pickle.dump(model, open(f'{self.model_path}/{model_name}.pkl', 'wb'))
        pickle.dump(self.dh, open(f'{self.model_path}/{model_name}_datahandler.pkl', 'wb'))
        json.dump(
            model.best_hyperparams,
            open(f'{self.model_path}/{model_name}_hparams.txt', 'w')
        )
    
    def load_model(self, predict_period, finetune=True):
        model_name = f'{self.model_type}_'
        model_name += f'predmonth_{predict_period}'
        if finetune is False:
            model_name += '_no_finetune'
        model = pickle.load(open(f'{self.model_path}/{model_name}.pkl', 'rb'))
        return model

    def run(self):
        # first datacrawling
        self.data_crawling()
        
        self.dh = DataHandler(
            train_dates=self.train_dates,
            test_dates=self.test_dates,
            scaling=self.scaling,
            model_type=self.model_type,
            pca=self.pca,
            correlation_value=self.correlation
        )
        labels, features = self.dh.load_data(
            target_name=self.target_name,
            selected_stocks=list(self.stock_dict.values())
        )

        for predict_period in range(1, self.predict_range+1):
            best_rmse = float('inf')
            best_model = None
            lookbacks = None
            if 'lookbacks' in self.hyperparams:
                lookbacks = self.hyperparams['lookbacks']

            # second data preprocess
            # TODO: uncertain for correlation
            # use feature_names as selected_stocks temporarily
            print('Start preprocessing data...')
            traindatasets, testdatasets = self.dh.data_preprocessing(
                labels=labels,
                features=features,
                predict_period=predict_period,
                lookbacks=lookbacks
            )
            print('Preprocessing data completes!')

            print(f'Start training the model of predicting {predict_period} months...')
            best_idx = 0
            for idx in tqdm(range(len(traindatasets))):
                # third train model
                model = FuturesModel(
                    train_dates=self.train_dates,
                    test_dates=self.test_dates,
                    target_name=self.target_name,
                    predict_period=predict_period,
                    model_type=self.model_type,
                    hyperparams=self.hyperparams
                )
                model.pipelining(traindatasets[idx], testdatasets[idx])
                if model.best_rmse < best_rmse:
                    best_rmse = model.best_rmse
                    best_model = model
                    best_idx = idx
            if lookbacks is not None:
                best_model.best_hyperparams['lookback'] = lookbacks[best_idx]
            # save no finetune model
            self.save_model(best_model, predict_period, False)
            # do finetuning DL model
            best_model.finetuning(testdatasets[best_idx])
            self.save_model(best_model, predict_period)
            self.save_dh(predict_period)
            self.models.append(best_model)
            print(f'Training the model of predicting {predict_period} months completes!')
    
    def inference(self, sdate, edate):
        pred_df = pd.DataFrame(columns=['Features Date', 'Predict Date', 'Predict'])
        for idx in range(len(self.models)):
            model = self.models[idx]
            lookback = None
            if 'lookback' in model.best_hyperparams:
                lookback = model.best_hyperparams['lookback']
            predict_period = idx + 1
            dataset = self.dh.split_for_inference(
                pred_sdate=sdate,
                pred_edate=edate,
                predict_period=predict_period,
                lookback=lookback
            )
            prediction, rmse = model.inference(dataset)
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
            print()
            pred_df = pred_df.append(df, ignore_index=True)
        print(pred_df)
        return pred_df

    def inference_by_load(self, sdate, edate, finetune_type=1):
        '''
        finetune type:
            0. no finetune
            1. finetune only in training
            2. finetune in training and finetune once when trading
            3. finetune in training and finetune constantly when trading
        '''
        pred_df = pd.DataFrame(columns=['Features Date', 'Predict Date', 'Predict'])
        for predict_period in range(1, self.predict_range+1):
            model = self.load_model(predict_period)
            if finetune_type == 0:
                model = self.load_model(predict_period, False)
            self.load_dh(predict_period)
            lookback = None
            if 'lookback' in model.best_hyperparams:
                lookback = model.best_hyperparams['lookback']
            
            ### modify here since diff. finetune types ###
            prediction = []
            rmse = 0
            if finetune_type <= 1:
                dataset = self.dh.split_for_inference(
                    pred_sdate=sdate,
                    pred_edate=edate,
                    predict_period=predict_period,
                    lookback=lookback
                )
                prediction, rmse = model.inference(dataset)
            else: # here sould coordinate with finetune_when_trade.py
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
                    dataset = self.dh.split_for_inference(
                        pred_sdate=trade_dates_month_begin[idx],
                        pred_edate=trade_dates_month_end[idx],
                        predict_period=predict_period,
                        lookback=lookback
                    )
                    pred, r = self.models[predict_period-1][idx].inference(dataset)
                    prediction += pred
                    rmse += r
                rmse /= len(trade_dates_month_begin)
            ### ### ### ### ### ### ### ### ### ###

            if self.scaling:
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
        print(pred_df)
        return pred_df
