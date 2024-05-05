import yfinance as yf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Tuple

class DataCrawler:
    def __init__(self, start_date, fill_missing=True):
        self.start_date = start_date
        self.fill_missing = fill_missing
        this_month = datetime.now().month
        this_month_year = datetime.now().year
        this_month_first_date = datetime(this_month_year, this_month, 1)
        self.end_date = (this_month_first_date - timedelta(days=1)).strftime('%Y-%m-%d')

    def crawl_daily_data(self, stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        ticker = yf.Ticker(stock_ticker)
        end_date_next_day = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        end_date_next_day = end_date_next_day.strftime('%Y-%m-%d') # use next day since ticker.history will not collect the end data
        df = ticker.history(start=start_date, end=end_date_next_day, interval='1d')
        df.index = df.index.date # only save date as index, no time period
        return df

    def store_business_day_data(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        daterange = pd.date_range(start=start_date, end=end_date, freq='B')
        df = df.reindex(daterange)
        return df

    def fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna(method='ffill', inplace=True) # previous data to next nan data
        df.fillna(method='bfill', inplace=True) # for nan 0-index data
        return df

    def meet_data_format(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(subset=['Open', 'Close'], inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index':'Date'}, inplace=True)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df

    def create_daily_data(self, stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.crawl_daily_data(stock_ticker, start_date, end_date)
        df = self.store_business_day_data(df, start_date, end_date)
        if self.fill_missing:
            self.fill_missing_data(df)
        df = self.meet_data_format(df)
        return df

    def get_daily_data(self, stock_name: str) -> pd.DataFrame:
        return pd.read_csv(f'./data/daily/{stock_name}.csv')

    def get_string_month(self, start_date: datetime) -> Tuple[str, str]:
        """include start date and next month date
        """
        sdate = datetime.strftime(start_date, '%Y-%m')
        edate = start_date + relativedelta(months=1)
        edate = datetime.strftime(edate, '%Y-%m')
        return sdate, edate

    def get_a_month_average_data(self, df_daily: pd.DataFrame, start_date: str, end_date: str) -> dict:
        df_a_month = df_daily[
            (df_daily['Date'] >= start_date)
            & (df_daily['Date'] < end_date)
        ].reset_index(drop=True)
        df_a_month = {
            'Date': df_a_month['Date'][0],
            'Open': df_a_month['Open'].mean(),
            'High': df_a_month['High'].mean(),
            'Low': df_a_month['Low'].mean(),
            'Close': df_a_month['Close'].mean(),
            'Volume': df_a_month['Volume'].mean(),
        }
        return df_a_month

    def create_monthly_average_data(self, df_daily: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        df_monthly = []
        daterange = pd.date_range(start=start_date, end=end_date, freq='MS') # get every month first date
        for i in range(len(daterange)):
            sdate, edate = self.get_string_month(daterange[i])
            df_a_month = self.get_a_month_average_data(df_daily, sdate, edate)
            df_monthly.append(df_a_month)
        df_monthly = pd.DataFrame(df_monthly, columns=list(df_daily.columns))
        return df_monthly

    def get_monthly_data(self, stock_name):
        return pd.read_csv(f'./data/monthly/{stock_name}.csv')

    def save_data(self, df, directory, stock_name):
        data_path = os.path.join(directory, f'{stock_name}.csv')
        df.to_csv(data_path, index=False)

    def crawl_data(self, stock_dict):
        '''
        stock_dict:
            keys are tickers
            values are stock names
        '''
        directory_daily = './data/daily'
        directory_monthly = './data/monthly'
        if not os.path.exists(directory_daily):
            os.makedirs(directory_daily)
        if not os.path.exists(directory_monthly):
            os.makedirs(directory_monthly)

        for ticker, stock_name in tqdm(stock_dict.items()):
            df_daily = self.create_daily_data(ticker, self.start_date, self.end_date)
            self.save_data(df_daily, directory_daily, stock_name)
            df_monthly = self.create_monthly_average_data(df_daily, self.start_date, self.end_date)
            self.save_data(df_monthly, directory_monthly, stock_name)

if __name__ == '__main__':
    stock_dict = {
        '^AORD': 'AORD',
        '^TWII': 'Taiex'
    }
    datacrawler = DataCrawler('2008-01-01', '2022-07-31')
    datacrawler.crawl_data(stock_dict)
