import os
import sys
import copy
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

TRAIN_STRAT = '2008-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2021-12-31'
MODEL = 'CRNN'
DO_ALL_MODEL = True
DO_ALL_YEAR = True
DL_models = ['LSTM', 'CRNN', 'TCN']

class Analyzer():
    def __init__(self, target_path):
        self.target_path = target_path
        self.trading_results = {}
        self.results_format = {
            'Trade Type': [],
            'Preprocess Type': [],
            'Earn': [],
            'Profit / Trade (Times)': [],
            'Accuracy': []
        }
        self.tradetime2df = {}
        self.best_trading_results = {}
    
    def run(self):
        self.analyze_trading_results()

    def select_contract(self, possible_contracts):
        selected_contract = possible_contracts[
            possible_contracts['Selection'] == 'Y'
        ].copy(deep=True)
        if len(selected_contract) == 0: # if this month does not trade
            selected_contract = possible_contracts.head(1).copy(deep=True)
        elif len(selected_contract) > 1:
            print("Warning: Selected contract trade more than 1 times in a month!")
        return selected_contract

    def hybrid_trading(self, selected_contract1, selected_contract2):
        earnings = 0
        if selected_contract1['Selection'].iat[0] == 'Y' and selected_contract2['Selection'].iat[0] == 'Y':
            if selected_contract1['Trading Type'].iat[0] == selected_contract2['Trading Type'].iat[0]:
                if selected_contract1['Contract Delivery Month'].iat[0] == selected_contract2['Contract Delivery Month'].iat[0]:
                    pass
                else:
                    earnings = selected_contract1['Profit'].iat[0] / 2 + selected_contract2['Profit'].iat[0] / 2
                    selected_contract1['Hybrid Selection'] = selected_contract1['Selection'].iat[0]
                    selected_contract2['Hybrid Selection'] = selected_contract2['Selection'].iat[0]
            else:
                earnings = selected_contract1['Profit'].iat[0] / 2 + selected_contract2['Profit'].iat[0] / 2
                selected_contract1['Hybrid Selection'] = selected_contract1['Selection'].iat[0]
                selected_contract2['Hybrid Selection'] = selected_contract2['Selection'].iat[0]
        else:
            selected_contract1['Hybrid Selection'] = selected_contract1['Selection'].iat[0]
            selected_contract2['Hybrid Selection'] = selected_contract2['Selection'].iat[0]

    def hybrid_DL_results(self):
        for trade_time, train2dict in self.best_trading_results:
            for train_time, model2df in train2dict:
                for model1, model2 in itertools.combinations(DL_models, 2):
                    df1 = model2df[model1]
                    df2 = model2df[model2]
                    df1_groups = df1.groupby('Trading Date')
                    df2_groups = df2.groupby('Trading Date')
                    hybrid_rows = list()
                    total_earnings = 0
                    for trading_date in df1_groups.groups.keys():
                        possible_contracts1 = df1_groups.get_group(trading_date)
                        selected_contract1 = self.select_contract(possible_contracts1)
                        possible_contracts2 = df2_groups.get_group(trading_date)
                        selected_contract2 = self.select_contract(possible_contracts2)
                        
                        selected_contract1['Model'] = model1

    def save_best_trading_results(self, model_type, train_time):
        for trade_time, df in self.tradetime2df.items():
            if trade_time not in self.best_trading_results:
                self.best_trading_results[trade_time] = dict()
            if train_time not in self.best_trading_results[trade_time]:
                self.best_trading_results[trade_time][train_time] = dict()
            self.best_trading_results[trade_time][train_time][model_type] = df

    def analyze_trading_results(self):
        for model_type in os.listdir(self.target_path):
            if DO_ALL_MODEL == False and MODEL != model_type:
                continue
            model_path = os.path.join(self.target_path, model_type)
            for train_time_dir in os.listdir(model_path):
                if DO_ALL_YEAR == False and train_time_dir != f'train_{TRAIN_STRAT}_{TRAIN_END}_test_{TEST_START}_{TEST_END}':
                    continue
                train_time_path = os.path.join(model_path, train_time_dir)
                self.trading_results = {}
                for dirs in os.listdir(train_time_path):
                    if dirs == 'trading_results':
                        results_path = os.path.join(train_time_path, dirs)
                        self.create_trading_results(results_path, 'No corr No scale No pca', 'finetune')
                    elif dirs == 'trading_results_no_finetune':
                        results_path = os.path.join(train_time_path, dirs)
                        self.create_trading_results(results_path, 'No corr No scale No pca', 'no_finetune')
                    elif dirs == 'trading_results_finetune_once':
                        results_path = os.path.join(train_time_path, dirs)
                        self.create_trading_results(results_path, 'No corr No scale No pca', 'finetune_once')
                    elif dirs == 'trading_results_finetune_constant':
                        results_path = os.path.join(train_time_path, dirs)
                        self.create_trading_results(results_path, 'No corr No scale No pca', 'finetune_constant')
                    elif dirs == 'pca':
                        pca_path = os.path.join(train_time_path, dirs)
                        results_path = os.path.join(pca_path, 'trading_results')
                        self.create_trading_results(results_path, 'PCA', 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(pca_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, 'PCA', 'no_finetune')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, 'PCA', 'finetune_once')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, 'PCA', 'finetune_constant')
                    elif dirs == 'scaling':
                        scaling_path = os.path.join(train_time_path, dirs)
                        results_path = os.path.join(scaling_path, 'trading_results')
                        self.create_trading_results(results_path, 'Scale', 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(scaling_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, 'Scale', 'no_finetune')
                            results_path = os.path.join(scaling_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, 'Scale', 'finetune_once')
                            results_path = os.path.join(scaling_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, 'Scale', 'finetune_constant')
                        
                        pca_path = os.path.join(scaling_path, 'pca')
                        results_path = os.path.join(pca_path, 'trading_results')
                        self.create_trading_results(results_path, 'Scale & PCA', 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(pca_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, 'Scale & PCA', 'no_finetune')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, 'Scale & PCA', 'finetune_once')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, 'Scale & PCA', 'finetune_constant')
                            # self.save_best_trading_results(model_type, train_time_dir)
                        
                    elif 'correlation' in dirs:
                        corr_path = os.path.join(train_time_path, dirs)

                        results_path = os.path.join(corr_path, 'trading_results')
                        self.create_trading_results(results_path, dirs, 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(corr_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, dirs, 'no_finetune')
                            results_path = os.path.join(corr_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, dirs, 'finetune_once')
                            results_path = os.path.join(corr_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, dirs, 'finetune_constant')

                        pca_path = os.path.join(corr_path, 'pca')
                        results_path = os.path.join(pca_path, 'trading_results')
                        self.create_trading_results(results_path, f'{dirs} & PCA', 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(pca_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, f'{dirs} & PCA', 'no_finetune')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, f'{dirs} & PCA', 'finetune_once')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, f'{dirs} & PCA', 'finetune_constant')

                        scaling_path = os.path.join(corr_path, 'scaling')
                        results_path = os.path.join(scaling_path, 'trading_results')
                        self.create_trading_results(results_path, f'{dirs} & scale', 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(scaling_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, f'{dirs} & scale', 'no_finetune')
                            results_path = os.path.join(scaling_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, f'{dirs} & scale', 'finetune_once')
                            results_path = os.path.join(scaling_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, f'{dirs} & scale', 'finetune_constant')

                        pca_path = os.path.join(scaling_path, 'pca')
                        results_path = os.path.join(pca_path, 'trading_results')
                        self.create_trading_results(results_path, f'{dirs} & scale & PCA', 'finetune')
                        if model_type in DL_models:
                            results_path = os.path.join(pca_path, 'trading_results_no_finetune')
                            self.create_trading_results(results_path, f'{dirs} & scale & PCA', 'no_finetune')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_once')
                            self.create_trading_results(results_path, f'{dirs} & scale & PCA', 'finetune_once')
                            results_path = os.path.join(pca_path, 'trading_results_finetune_constant')
                            self.create_trading_results(results_path, f'{dirs} & scale & PCA', 'finetune_constant')

                analysis_path = os.path.join(train_time_path, 'analysis')
                for finetune_type, results in self.trading_results.items():
                    for key, value in results.items():
                        self.create_analysis(analysis_path, key, value, finetune_type)

    def create_trading_results(self, results_path, preprocess_type, finetune_type):
        if os.path.isdir(results_path):
            for trade_time_dir in os.listdir(results_path):
                trade_time_path = os.path.join(results_path, trade_time_dir)
                if os.path.isdir(trade_time_path):
                    if finetune_type not in self.trading_results:
                        self.trading_results[finetune_type] = {}
                    if trade_time_dir not in self.trading_results[finetune_type]:
                        self.trading_results[finetune_type][trade_time_dir] = copy.deepcopy(self.results_format)
                    for dir_path, dir_names, file_names in os.walk(trade_time_path):
                        for f in file_names:
                            file_path = os.path.join(dir_path, f)
                            df = pd.read_csv(file_path)
                            earn = df['Total'][0]
                            trade_times = len(df[df['Selection'] == 'Y'])
                            profit_times = len(df[df['Profit'] > 0])
                            profit_div_trade = f'{profit_times}/{trade_times}'
                            accuracy = 0 if trade_times == 0 else profit_times / trade_times * 100
                            trade_type = f.replace('_contracts', '')
                            if '_tolerance' in trade_type:
                                trade_type = trade_type.replace('_tolerance', '')
                            trade_type = trade_type.replace('.csv', '')
                            if trade_type == 'longorshort' and finetune_type == 'finetune_constant' and preprocess_type == 'Scale & PCA':
                                self.tradetime2df[trade_time_dir] = df
                            self.trading_results[finetune_type][trade_time_dir]['Trade Type'].append(trade_type)
                            self.trading_results[finetune_type][trade_time_dir]['Preprocess Type'].append(preprocess_type)
                            self.trading_results[finetune_type][trade_time_dir]['Earn'].append(earn)
                            self.trading_results[finetune_type][trade_time_dir]['Profit / Trade (Times)'].append(profit_div_trade)
                            self.trading_results[finetune_type][trade_time_dir]['Accuracy'].append(accuracy)
        else:
            print(f'{results_path} not found!!!')
            # sys.exit(1)

    def create_analysis(self, analysis_path, trade_time, trade_result, finetune_type):
        analysis_time_path = os.path.join(analysis_path, finetune_type, trade_time)
        os.makedirs(analysis_time_path, exist_ok=True)
        df = pd.DataFrame(trade_result)
        df = df.set_index(keys=['Trade Type', 'Preprocess Type'])
        df_earn = df['Earn']
        df_earn = df_earn.unstack(level=-1)
        df_earn_no_short = df_earn[['long' in i for i in df_earn.index]]
        fig_earn_no_short = df_earn_no_short.plot(kind='bar', figsize=(12, 7), rot=0, ylabel='Point').get_figure()
        fig_earn_no_short.savefig(f'{analysis_time_path}/earn_no_short.png')
        fig_earn = df_earn.plot(kind='bar', figsize=(12, 9), rot=75, ylabel='Point').get_figure()
        fig_earn.savefig(f'{analysis_time_path}/earn.png')
        df_earn.to_csv(f'{analysis_time_path}/earn.csv')
        df_profit_div_trade = df['Profit / Trade (Times)']
        df_profit_div_trade = df_profit_div_trade.unstack(level=-1)
        df_profit_div_trade.rename(columns=lambda x: x.replace('correlation', 'corr') if 'correlation' in x else x, inplace=True)
        plt.figure(figsize=(12, 4))
        ax = plt.axes(frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        tbl = pd.plotting.table(ax, df_profit_div_trade, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)
        plt.savefig(f'{analysis_time_path}/profit_div_trade.png')
        df_profit_div_trade.to_csv(f'{analysis_time_path}/profit_div_trade.csv')
        df_accuracy = df['Accuracy']
        df_accuracy = df_accuracy.unstack(level=-1)
        fig_accuracy = df_accuracy.plot(kind='bar', figsize=(12, 7), rot=0, ylabel='Accuracy(%)').get_figure()
        fig_accuracy.savefig(f'{analysis_time_path}/accuracy.png')
        df_accuracy.to_csv(f'{analysis_time_path}/accuracy.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', nargs='?', type=str, default='Taiex', help='choose target for analysis')
    parser.add_argument('-m', '--model', nargs='*', type=str, default=None, help='choose model types for analysis')
    parser.add_argument('-y', '--year', nargs='?', type=int, default=None, help='choose year for analysis')
    args = parser.parse_args()

    target = args.target
    target_path = './models/target_' + target + '/'

    if args.model != None:
        DO_ALL_MODEL = False
        MODEL = args.model
    if args.year != None:
        DO_ALL_YEAR = False
        TRAIN_STRAT = '2008-01-01'
        TRAIN_END = f'{args.year-2}-12-31'
        TEST_START = f'{args.year-1}-01-01'
        TEST_END = f'{args.year-1}-12-31'

    if os.path.exists(target_path):
        analyzer = Analyzer(target_path)
        analyzer.run()
    else:
        print(f'{target_path} not found!!!')
        sys.exit(1)