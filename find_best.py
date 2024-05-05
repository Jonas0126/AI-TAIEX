import os
import sys
import json
import copy
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class Searcher():
    def __init__(self, model_path):
        self.model_path = model_path
        self.this_model_best_results = {
            'Trade Type': '',
            'Preprocess Type': '',
            'Best Earn': 0
        }
        self.all_results = []

    def run(self):
        if os.path.isdir(self.model_path):
            for time_range_f in os.listdir(self.model_path):
                self.this_model_best_results = {
                    'Trade Type': '',
                    'Preprocess Type': '',
                    'Best Earn': 0
                }
                time_range_path = os.path.join(self.model_path, time_range_f)
                if os.path.isdir(time_range_path):
                    for dirs in os.listdir(time_range_path):
                        if dirs == 'trading_results':
                            results_path = os.path.join(time_range_path, dirs)
                            self.get_trading_results(results_path, 'No corr No scale')
                        elif dirs == 'scaling':
                            scaling_path = os.path.join(time_range_path, dirs)
                            results_path = os.path.join(scaling_path, 'trading_results')
                            self.get_trading_results(results_path, 'Scale')
                        elif 'correlation' in dirs:
                            corr_path = os.path.join(time_range_path, dirs)
                            results_path = os.path.join(corr_path, 'trading_results')
                            self.get_trading_results(results_path, dirs)
                            scaling_path = os.path.join(corr_path, 'scaling')
                            results_path = os.path.join(scaling_path, 'trading_results')
                            self.get_trading_results(results_path, f'{dirs} & scale')
                    best_model_path = os.path.join(time_range_path, 'best')
                    self.create_best(best_model_path)
    
    def get_trading_results(self, results_path, preprocess_type):
        if os.path.isdir(results_path):
            trade_time_dir = os.listdir(results_path)[0]
            trade_time_path = os.path.join(results_path, trade_time_dir)
            if os.path.isdir(trade_time_path):
                for dir_path, dir_names, file_names in os.walk(trade_time_path):
                    for f in file_names:
                        file_path = os.path.join(dir_path, f)
                        df = pd.read_csv(file_path)
                        earn = df['Total'][0]
                        trade_type = f.replace('_contracts', '')
                        if '_tolerance' in trade_type:
                            trade_type = trade_type.replace('_tolerance', '')
                        trade_type = trade_type.replace('.csv', '')
                        if earn > self.this_model_best_results['Best Earn']:
                            self.this_model_best_results['Best Earn'] = earn
                            self.this_model_best_results['Trade Type'] = trade_type
                            self.this_model_best_results['Preprocess Type'] = preprocess_type
                        result = {
                            'Trade Time': trade_time_dir,
                            'Trade Type': trade_type,
                            'Preprocess Type': preprocess_type,
                            'Earn': earn
                        }
                        self.all_results.append(result)
        else:
            # print(f'{results_path} not found!!!')
            # sys.exit(1)
            pass
    
    def create_best(self, best_model_path):# same model, same time, diff preprocess
        os.makedirs(best_model_path, exist_ok=True)
        json.dump(
            self.this_model_best_results,
            open(f'{best_model_path}/best.txt', 'w')
        )

def diff_model_same_preprocess_analysis(all_results, target_path):
    same_t_same_pp_results = {}
    temp_format = {
        'Trade Type': [],
        'Model Type': [],
        'Earn': [],
    }
    for model, r_list in all_results.items():# r_list is a list of results
        for r in r_list:
            if r['Trade Time'] not in same_t_same_pp_results.keys():
                same_t_same_pp_results[r['Trade Time']] = {}
                same_t_same_pp_results[r['Trade Time']][r['Preprocess Type']] = copy.deepcopy(temp_format)
            elif r['Preprocess Type'] not in same_t_same_pp_results[r['Trade Time']].keys():
                same_t_same_pp_results[r['Trade Time']][r['Preprocess Type']] = copy.deepcopy(temp_format)
            same_t_same_pp_results[r['Trade Time']][r['Preprocess Type']]['Trade Type'].append(r['Trade Type'])
            same_t_same_pp_results[r['Trade Time']][r['Preprocess Type']]['Model Type'].append(model)
            same_t_same_pp_results[r['Trade Time']][r['Preprocess Type']]['Earn'].append(r['Earn'])

    for trade_time in same_t_same_pp_results.keys():
        analysis_path = os.path.join(target_path, 'analysis')
        trade_time_path = os.path.join(analysis_path, trade_time)
        for pp_type in same_t_same_pp_results[trade_time].keys():
            pp_path = os.path.join(trade_time_path, pp_type)
            os.makedirs(pp_path, exist_ok=True)
            trade_results = same_t_same_pp_results[trade_time][pp_type]
            df = pd.DataFrame(trade_results)
            df = df.set_index(keys=['Trade Type', 'Model Type'])
            df_earn = df['Earn']
            df_earn = df_earn.unstack(level=-1)
            fig_earn = df_earn.plot(kind='bar', figsize=(12, 7), rot=0, ylabel='Point').get_figure()
            fig_earn.savefig(f'{pp_path}/earn.png')
            df_earn.to_csv(f'{pp_path}/earn.csv')
            plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', nargs='?', type=str, default='Taiex', help='choose target for analysis')
    args = parser.parse_args()

    target = args.target
    target_path = './models/target_' + target + '/'

    if os.path.exists(target_path):
        all_results = {}
        for model_f in tqdm(os.listdir(target_path)):
            if model_f == 'analysis':
                continue
            model_path = os.path.join(target_path, model_f)
            searcher = Searcher(model_path)
            searcher.run()
            all_results[model_f] = searcher.all_results
        diff_model_same_preprocess_analysis(all_results, target_path)
        print(f'Successfully Find all best model!!!')
    else:
        print(f'{target_path} not found!!!')
        sys.exit(1)