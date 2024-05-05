import os
import sys
import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

TRAIN_STRAT = '2008-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2021-12-31'
MODEL = 'CRNN'
DO_ALL_MODEL = True
DO_ALL_YEAR = True

class DataVisualizer():
    def __init__(self, target_path, contract_name) -> None:
        """
        time_dict = {
            trade time 1: {
                train time 1: {model1_dict, model2_dict},
                train time 2: {model2_dict, model3_dict}
            },
            trade time 2: {
                ...
            }
        }
        """
        self.target_path = target_path
        self.time_dict = dict()
        self.contract_name = contract_name
        self.contract_df = pd.read_csv(f'./contracts/{contract_name}.csv')
        self.contract_df_monthly = pd.read_csv(f'./data/monthly/{contract_name}.csv')

    def visualize_all_target_results(self):
        for trade_time in self.time_dict.keys():
            for train_time in self.time_dict[trade_time].keys():
                max_sharpe_ratio = dict()
                min_sharpe_ratio = dict()
                for model_type, tradetype2df_visual in self.time_dict[trade_time][train_time].items():
                    for trade_type, df in tradetype2df_visual.items():
                        if trade_type in max_sharpe_ratio.keys():
                            max_sharpe_ratio[trade_type] = max(max_sharpe_ratio[trade_type], round(df[np.isfinite(df['Sharpe Ratio'])]['Sharpe Ratio'].max(), 3))
                            min_sharpe_ratio[trade_type] = min(min_sharpe_ratio[trade_type], round(df[np.isfinite(df['Sharpe Ratio'])]['Sharpe Ratio'].min(), 3))
                        else:
                            max_sharpe_ratio[trade_type] = round(df[np.isfinite(df['Sharpe Ratio'])]['Sharpe Ratio'].max(), 3)
                            min_sharpe_ratio[trade_type] = round(df[np.isfinite(df['Sharpe Ratio'])]['Sharpe Ratio'].min(), 3)
                for model_type, tradetype2df_visual in self.time_dict[trade_time][train_time].items():
                    visual_path = f'./visualization/{self.contract_name}/{trade_time}/{train_time}/{model_type}'
                    os.makedirs(visual_path, exist_ok=True)
                    self.visualize_results(visual_path, tradetype2df_visual, max_sharpe_ratio, min_sharpe_ratio)


    def get_all_target_results(self):
        for model_dir in os.listdir(self.target_path):
            if DO_ALL_MODEL == False and MODEL != model_dir:
                continue
            model_path = os.path.join(self.target_path, model_dir)
            for train_time_dir in os.listdir(model_path):
                if DO_ALL_YEAR == False and train_time_dir != f'train_{TRAIN_STRAT}_{TRAIN_END}_test_{TEST_START}_{TEST_END}':
                    continue
                if model_dir == 'LR' or model_dir == 'XGB':
                    results_path = os.path.join(model_path, train_time_dir, 'scaling', 'pca', 'trading_results') # only support scaling & pca
                else:
                    results_path = os.path.join(model_path, train_time_dir, 'scaling', 'pca', 'trading_results_finetune_constant') # only support scaling & pca
                if os.path.exists(results_path):
                    self.get_trading_results(results_path, train_time_dir, model_dir)
    
    def get_trading_results(self, results_path, train_time_dir, model_type):
        for trade_time_dir in os.listdir(results_path):
            if trade_time_dir not in self.time_dict:
                self.time_dict[trade_time_dir] = dict()
            if train_time_dir not in self.time_dict[trade_time_dir]:
                self.time_dict[trade_time_dir][train_time_dir] = dict()
            # TODO: support different trading types
            tradetype2df = dict()
            trade_time_path = os.path.join(results_path, trade_time_dir)
            for dir_path, dir_names, file_names in os.walk(trade_time_path):
                for f in file_names:
                    trade_type = f.replace('_contracts', '')
                    if '_tolerance' in trade_type:
                        trade_type = trade_type.replace('_tolerance', '')
                    trade_type = trade_type.replace('.csv', '')
                    tradetype2df[trade_type] = self.choose_data_to_visualize(trade_type, pd.read_csv(os.path.join(dir_path, f)))
            self.time_dict[trade_time_dir][train_time_dir][model_type] = self.calculate_results(tradetype2df)

    def choose_data_to_visualize(self, trade_type, df) -> pd.DataFrame:
        df_groups = df.groupby('Trading Date')
        selected_rows = list()
        for group_name in df_groups.groups.keys():
            selected_df = df_groups.get_group(group_name)
            selected_row = selected_df[
                selected_df['Selection'] == 'Y'
            ].copy(deep=True)
            if len(selected_row) == 0: # if this month does not trade
                if 'long' in trade_type:
                    selected_row = selected_df.sort_values(by=['Difference'], ascending=False).head(1).copy(deep=True)
                elif 'short' in trade_type:
                    selected_row = selected_df.sort_values(by=['Difference']).head(1).copy(deep=True)
            elif len(selected_row) > 1:
                print("Currently no support trading more than 1 times in a month!")
                sys.exit(1)
            monthly_avg = self.contract_df_monthly[
                self.contract_df_monthly['Date'].str.match(group_name)
            ]
            if len(monthly_avg) > 1 or len(monthly_avg) <= 0:
                print("Monthly average wrong!")
                print(group_name)
                print(monthly_avg)
                sys.exit(1)
            selected_row['Monthly Open AVG'] = monthly_avg['Open'].iat[0]
            if selected_row['Selection'].iat[0] == 'Y':
                trade2delivery_df = self.contract_df[
                    (self.contract_df['Trading Date'] >= selected_row['Trading Date'].iat[0])
                    & (self.contract_df['Delivery Month'] == selected_row['Contract Delivery Month'].iat[0])
                ].sort_values(by=['Trading Date']).reset_index(drop=True)
                selected_row['Low'] = trade2delivery_df['Low'].min()
                selected_row['High'] = trade2delivery_df['High'].max()
                selected_row['Monthly Open AVG'] = trade2delivery_df['Open'].mean()
                selected_row['Close'] = trade2delivery_df['Close'].tail(1).iat[0]
            else:
                selected_row['Low'] = monthly_avg['Open'].iat[0]
                selected_row['High'] = monthly_avg['Open'].iat[0]
                selected_row['Close'] = monthly_avg['Open'].iat[0]
            selected_rows.append(selected_row)
        return pd.concat(selected_rows, ignore_index=True).sort_values(by=['Trading Date'])

    def calculate_results(self, tradetype2df):
        tradetype2df_visual = dict()
        for trade_type, df in tradetype2df.items():
            df_visual = df.set_index(keys=['Trading Date'])
            df_visual['Cumulative Profit AVG'] = df_visual['Profit'].rolling(len(df_visual), min_periods=2).mean()
            df_visual['Cumulative STD'] = df_visual['Profit'].rolling(len(df_visual), min_periods=2).std()
            df_visual['Sharpe Ratio'] = df_visual['Cumulative Profit AVG'] / df_visual['Cumulative STD']
            df_visual['Error Ratio'] = abs(df_visual['Predict'] - df_visual['Monthly Open AVG']) / df_visual['Monthly Open AVG']
            tradetype2df_visual[trade_type] = df_visual
        return tradetype2df_visual

    def visualize_results(self, visual_path, tradetype2df_visual, max_sharpe_ratio, min_sharpe_ratio):
        for trade_type, df_visual in tradetype2df_visual.items():
            fig, ax = plt.subplots(figsize=(12, 10))
            p1 = ax.plot(df_visual.index, df_visual['Open'], '-bo', label='Open')
            p2 = ax.plot(df_visual.index, df_visual['Predict'], '-yo', label='Predict')
            p3 = ax.bar(df_visual.index, df_visual['Profit'], color='gray', label='Profit')
            # p4 = ax2.plot(df_visual.index, df_visual['Sharpe Ratio'], '-ro', label='Sharpe Ratio')
            p5 = ax.plot(df_visual.index, df_visual['Monthly Open AVG'], '-go', label='Monthly Open AVG')
            p6 = ax.plot(df_visual.index, df_visual['High'], '-co', label='High')
            p7 = ax.plot(df_visual.index, df_visual['Low'], '-mo', label='Low')
            p8 = ax.plot(df_visual.index, df_visual['Close'], color='pink', linestyle='solid', marker='o', label='Close')
            ax.set_ylabel('Index Point')
            ax.set_xlabel('Date')
            ps = p1+p2+p5+p6+p7+p8
            labs = [p.get_label() for p in ps]
            labs.append('Profit')
            ps.append(plt.Rectangle((0,0),1,1, color='gray'))
            ax.legend(ps, labs)
            fig.savefig(os.path.join(visual_path, f'{trade_type}.png'))
            plt.close()
            # rmse and sharpe ratio
            fig, ax = plt.subplots(figsize=(12, 10))
            ax2 = ax.twinx()
            p1 = ax.bar(df_visual.index, df_visual['Sharpe Ratio'], color='gray', label='Sharpe Ratio')
            p2 = ax2.plot(df_visual.index, df_visual['Error Ratio'], '-ro', label='Error Ratio')
            # yticks2_start, yticks2_end = ax2.get_ylim()
            # other offset
            # yticks2_start, yticks2_end = max_sharpe_ratio[trade_type], min_sharpe_ratio[trade_type]
            # offset = yticks2_end - yticks2_start
            # yticks2_start -= offset
            # yticks2_end += offset
            # ax2.set_yticks([round(v, 2) for v in np.linspace(yticks2_start, yticks2_end, 5)])
            ax2.yaxis.label.set_color('red')
            ax2.spines.right.set_color('red')
            ax2.tick_params(axis='y', colors='red')
            ax.set_ylabel('Sharpe Ratio')
            ax2.set_ylabel('Error Ratio')
            ax.set_xlabel('Date')
            ps = p2
            labs = [p.get_label() for p in ps]
            labs.append('Sharpe Ratio')
            ps.append(plt.Rectangle((0,0),1,1, color='gray'))
            ax.legend(ps, labs)
            fig.savefig(os.path.join(visual_path, f'rmse_{trade_type}.png'))
            plt.close()

    def get_trade_time_target_monthly_open_data(self) -> dict:
        """
        currently no use
        trade_time2target_data = {
            trade time 1: df,
            trade time 2: df
        }
        df:   'MS' 'AVG'
        date1  1     2
        date2  3     4
        """
        trade_time2target_data = dict()
        for trade_time in self.time_dict.keys():
            trade_start, trade_end = trade_time.split('_', 1)
            daterange = pd.date_range(start=trade_start, end=trade_end, freq='MS').strftime('%Y-%m-%d')
            ms_list = []
            avg_list = []
            for i in range(len(daterange)):
                a_month_df = self.contract_df[
                    (self.contract_df['Date'] >= daterange[i])
                    & (self.contract_df['Date'] <= f'{daterange[i][:-2]}31')
                ]
                ms_list.append(a_month_df['Open'][0])
                avg_list.append(a_month_df['Open'].mean())
            data_dict = {'MS': ms_list, 'AVG': avg_list}
            trade_time2target_data[trade_time] = pd.DataFrame(data_dict, index=daterange)
        return trade_time2target_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', nargs='?', type=str, default='Taiex', help='choose target for data visualization')
    parser.add_argument('-m', '--model', nargs='?', type=str, default=None, help='choose model for data visualization')
    parser.add_argument('-y', '--year', nargs='?', type=int, default=None, help='choose year for data visualization')
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
        if target == 'Taiex':
            dv = DataVisualizer(target_path, 'MTX')
        elif target == 'SP500':
            dv = DataVisualizer(target_path, 'E-miniSP500')
        else:
            print('not support this target!!!')
            sys.exit(1)
        dv.get_all_target_results()
        dv.visualize_all_target_results()
    else:
        print(f'{target_path} not found!!!')
        sys.exit(1)