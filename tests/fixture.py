from futuresoperations.preparing import DataCrawler
import pandas as pd
import pytest

@pytest.fixture()
def test_datacrawler():
    return DataCrawler('2020-01-01')

@pytest.fixture()
def test_df_2020_01():
    idx_b = pd.date_range('2020-01-01 00:00-05:00', '2020-01-31 00:00-05:00', freq='B')
    idx = idx_b.drop(['2020-01-01', '2020-01-20']) # U.S. holiday
    d = {'Open': range(len(idx)), 'Close': range(len(idx))}
    test_df = pd.DataFrame(data=d, index=idx) # simulate yfinance dataframe
    return test_df

@pytest.fixture()
def test_df_daily_2020_3_4():
    idx_b = pd.date_range('2020-03-01', '2020-04-30', freq='B')
    d = {
        'Date': idx_b.strftime('%Y-%m-%d'),
        'Open': range(len(idx_b)),
        'High': range(0, len(idx_b)*2, 2),
        'Low': range(0, -len(idx_b), -1),
        'Close': range(1, len(idx_b)+1),
        'Volume': range(100, len(idx_b)+100)
    }
    test_df = pd.DataFrame(data=d) # simulate formated dataframe
    return test_df