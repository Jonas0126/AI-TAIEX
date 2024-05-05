from futuresoperations.preparing import DataCrawler
from .fixture import test_datacrawler, test_df_2020_01, test_df_daily_2020_3_4
from datetime import datetime
import pandas as pd
import copy
import os

def test_crawl_daily_data(test_datacrawler):
    test_stock_ticker = '^GSPC' # S&P 500
    test_start_date = '2020-01-02'
    test_end_date = '2020-01-03'
    test_df = test_datacrawler.crawl_daily_data(
        stock_ticker=test_stock_ticker,
        start_date=test_start_date,
        end_date=test_end_date
    )
    assert isinstance(test_df, pd.DataFrame)
    assert all(OHLCV in test_df.columns for OHLCV in ['Open', 'High', 'Low', 'Close', 'Volume'])
    assert round(test_df['Open'][0], 2) == 3244.67
    assert round(test_df['High'][0], 2) == 3258.14
    assert round(test_df['Low'][0], 2) == 3235.53
    assert round(test_df['Close'][0], 2) == 3257.85
    assert test_df['Volume'][0] == 3459930000

def test_store_business_day_data(test_datacrawler, test_df_2020_01):
    test_df = copy.deepcopy(test_df_2020_01)
    test_df = test_datacrawler.store_business_day_data(test_df, '2020-01-01', '2020-01-31')
    idx_b = pd.date_range('2020-01-01', '2020-01-31', freq='B')
    assert test_df.index.equals(idx_b)
    assert test_df.loc[['2020-01-01', '2020-01-20'], 'Open'].isnull().sum() == 2

def test_fill_missing_data(test_datacrawler):
    d = {'col1': [1, 2, float('nan')], 'col2': [3, float('nan'), 4], 'col3': [float('nan'), 5, 6]}
    test_df = pd.DataFrame(data=d)
    test_df = test_datacrawler.fill_missing_data(test_df)
    assert test_df.isnull().sum().sum() == 0

def test_meet_data_format(test_datacrawler, test_df_2020_01):
    test_df = copy.deepcopy(test_df_2020_01)
    test_df = test_datacrawler.meet_data_format(test_df)
    assert 'Date' in test_df.columns
    assert test_df.index.inferred_type == 'integer'
    assert test_df['Date'][0] == test_df_2020_01.index[0].strftime('%Y-%m-%d')

def test_create_daily_data(test_datacrawler):
    test_df = test_datacrawler.create_daily_data('^GSPC', '2020-03-01', '2020-04-30')
    assert len(test_df) == 44

def test_get_string_month(test_datacrawler):
    ans_start = '2021-04'
    ans_next = '2021-05'
    test_start = datetime.strptime('2021-04-01', '%Y-%m-%d')
    test_ans_start, test_ans_end = test_datacrawler.get_string_month(test_start)
    assert test_ans_start == ans_start
    assert test_ans_end == ans_next

def test_get_a_month_average_data(test_datacrawler, test_df_daily_2020_3_4):
    test_df = copy.deepcopy(test_df_daily_2020_3_4)
    test_dict = test_datacrawler.get_a_month_average_data(test_df, '2020-03-01', '2020-04-01')
    assert test_dict['Date'] == '2020-03-02'
    assert test_dict['Open'] == 10.5
    assert test_dict['High'] == 21.0
    assert test_dict['Low'] == -10.5
    assert test_dict['Close'] == 11.5
    assert test_dict['Volume'] == 110.5

def test_create_monthly_average_data(test_datacrawler, test_df_daily_2020_3_4):
    test_df = copy.deepcopy(test_df_daily_2020_3_4)
    test_df = test_datacrawler.create_monthly_average_data(test_df, '2020-03-01', '2020-04-30')
    assert test_df['Date'][0] == '2020-03-02'
    assert test_df['Open'][0] == 10.5
    assert test_df['High'][0] == 21.0
    assert test_df['Low'][0] == -10.5
    assert test_df['Close'][0] == 11.5
    assert test_df['Volume'][0] == 110.5
    assert test_df['Date'][1] == '2020-04-01'
    assert test_df['Open'][1] == 32.5
    assert test_df['High'][1] == 65.0
    assert test_df['Low'][1] == -32.5
    assert test_df['Close'][1] == 33.5
    assert test_df['Volume'][1] == 132.5

def test_crawl_data(test_datacrawler):
    stock_dict = {'^GSPC': 'SP500'}
    test_datacrawler.crawl_data(stock_dict)
    assert os.path.exists(os.path.join('data', 'daily', 'SP500.csv'))
    assert os.path.exists(os.path.join('data', 'monthly', 'SP500.csv'))