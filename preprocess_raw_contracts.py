from datetime import datetime
import yfinance as yf
import pandas as pd
import argparse
from math import ceil
import os
import sys

def calculate_month_offsets(trading_date, contract_name):
    if contract_name == 'E-miniSP500':
        if trading_date.month % 3 == 0:
            week = week_of_month(trading_date)
            if week > 3 or (week == 3 and trading_date.isoweekday() > 5):
                return 3
            else:
                return 0
        elif trading_date.month % 3 == 1:
            return 2
        else:
            return 1
    else:
        return 0

def week_of_month(dt):
    first_day = dt.replace(day=1)
    first_day_adjusted_weekday = first_day.isoweekday() % 7
    dom = dt.day
    adjusted_dom = dom + first_day_adjusted_weekday
    return int(ceil(adjusted_dom/7.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='give a contract name')
    args = parser.parse_args()

    start_year = 2016
    this_year = 2023
    this_month = 1
    contract_name = args.target
    target_path = f'./raw_contracts/{contract_name}/'
    all_contracts = None
    no_year_contracts = []
    new_contract_path = f'./contracts/{contract_name}.csv'

    if contract_name == 'MTX':
        # for past year contracts
        contract_list = []
        for year in range(start_year, this_year):
            target = target_path + str(year) + '_fut.csv'
            print(year)
            if os.path.isfile(target):
                df = pd.read_csv(target, index_col=False)
                df = df[df['契約']=='MTX']
                if '交易時段' in df: # 2016 don't have 交易時段 column
                    df = df[df['交易時段']=='一般']
                df = df[~df['到期月份(週別)'].str.contains('W')]
                df = df[~df['到期月份(週別)'].str.contains('/')]
                df['到期月份(週別)'] = df['到期月份(週別)'].str.replace("  ", "")
                df['到期月份(週別)'] = pd.to_datetime(df['到期月份(週別)'], format='%Y%m')
                df['到期月份(週別)'] = df['到期月份(週別)'].dt.strftime('%Y-%m')
                df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y/%m/%d')
                df['交易日期'] = df['交易日期'].dt.strftime('%Y-%m-%d')
                df = df.reset_index(drop=True)
                contract_list.append(df.iloc[:, :16])
                all_contracts = pd.concat(contract_list, ignore_index=True)
            else:
                no_year_contracts.append(year)

        # for no year contracts
        # for year in no_year_contracts:
        #     for i in range(1, 13):
        #         if i - 10 < 0:
        #             target = f'{target_path}{year}0{i}.csv'
        #         else:
        #             target = f'{target_path}{year}{i}.csv'
        #         df = pd.read_csv(target, index_col=False)
        #         df = df[df['契約']=='MTX']
        #         if '交易時段' in df: # 2016 don't have 交易時段 column
        #             df = df[df['交易時段']=='一般']
        #         df = df[~df['到期月份(週別)'].str.contains('W')]
        #         df = df[~df['到期月份(週別)'].str.contains('/')]
        #         df['到期月份(週別)'] = df['到期月份(週別)'].str.replace("  ", "")
        #         df['到期月份(週別)'] = pd.to_datetime(df['到期月份(週別)'], format='%Y%m')
        #         df['到期月份(週別)'] = df['到期月份(週別)'].dt.strftime('%Y-%m')
        #         df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y/%m/%d')
        #         df['交易日期'] = df['交易日期'].dt.strftime('%Y-%m-%d')
        #         df = df.reset_index(drop=True)
        #         all_contracts = all_contracts.append(df.iloc[:, :16])

        # # for this year contracts
        # for i in range(1, this_month):
        #     if i - 10 < 0:
        #         target = f'{target_path}{this_year}0{i}.csv'
        #     else:
        #         target = f'{target_path}{this_year}{i}.csv'
        #     df = pd.read_csv(target, index_col=False)
        #     df = df[df['契約']=='MTX']
        #     if '交易時段' in df: # 2016 don't have 交易時段 column
        #         df = df[df['交易時段']=='一般']
        #     df = df[~df['到期月份(週別)'].str.contains('W')]
        #     df = df[~df['到期月份(週別)'].str.contains('/')]
        #     df['到期月份(週別)'] = df['到期月份(週別)'].str.replace("  ", "")
        #     df['到期月份(週別)'] = pd.to_datetime(df['到期月份(週別)'], format='%Y%m')
        #     df['到期月份(週別)'] = df['到期月份(週別)'].dt.strftime('%Y-%m')
        #     df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y/%m/%d')
        #     df['交易日期'] = df['交易日期'].dt.strftime('%Y-%m-%d')
        #     df = df.reset_index(drop=True)
        #     all_contracts = all_contracts.append(df.iloc[:, :16])

        all_contracts = all_contracts.sort_values(by=['交易日期'])
        all_contracts.to_csv(f'{target_path}{contract_name}_all.csv', index=False)

        # to needed contract
        df = pd.read_csv(f'{target_path}{contract_name}_all.csv', index_col=False)
        df = df[['交易日期', '到期月份(週別)', '開盤價', '最高價', '最低價', '收盤價', '成交量']]
        new_column_names = {
            '交易日期': 'Trading Date', 
            '到期月份(週別)': 'Delivery Month',
            '開盤價': 'Open',
            '最高價': 'High',
            '最低價': 'Low',
            '收盤價': 'Close',
            '成交量': 'Volume'
        }
        df.rename(columns=new_column_names, inplace=True)
        df.sort_values(by=['Trading Date', 'Delivery Month'])
        df.to_csv(new_contract_path, index=False)
        df_monthly = df[
            df['Trading Date'].str.slice(stop=7) == df['Delivery Month']
        ]
        df_monthly = df_monthly.groupby('Delivery Month').mean()
        print(df_monthly.head(100).to_string())
        df_monthly.index.name = 'Date'
        df_monthly.reset_index(inplace=True)
        df_monthly.to_csv(f'./data/monthly/{contract_name}.csv', index=False)
    elif contract_name == 'E-miniSP500':
        ticker = yf.Ticker('ES=F')
        df = ticker.history(start='2016-01-01', end=f'{this_year}-01-01', interval='1d')
        df = df.reset_index().rename(columns={'Date': 'Trading Date'})
        df['Add Month Offsets'] = df['Trading Date'].apply(lambda x: calculate_month_offsets(x, 'E-miniSP500'))
        df['Delivery Month'] = df['Trading Date'].dt.to_period('M') + df['Add Month Offsets'].apply(pd.offsets.MonthEnd)
        df['Trading Date'] = df['Trading Date'].dt.strftime('%Y-%m-%d')
        df['Delivery Month'] = df['Delivery Month'].dt.strftime('%Y-%m')
        df.to_csv(new_contract_path, index=False)
    else:
        print(f'No contract named "{contract_name}"!!!')
