import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

class ContractGenerator:
    def __init__(self, start_date, end_date, fixed_months,
                 total_front_month=1, predict_range=3):
        '''
        total_front_month means total number of front month
        '''
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_months = fixed_months
        self.total_front_month = total_front_month
        self.predict_range = predict_range
        
    def calculate_contract_dates(self):
        all_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='MS'
        )
        # take out the day of date
        all_dates = all_dates.strftime('%Y-%m')

        possible_contract = pd.DataFrame(columns=['Date', 'Possible Contract Date'])
        possible_contract['Date'] = all_dates
        for i in range(len(possible_contract)):
            possible_contract_dates = []
            now_date = possible_contract['Date'][i]
            now_date = datetime.strptime(now_date, '%Y-%m')
            for j in range(self.total_front_month):
                if len(possible_contract_dates) >= self.predict_range:
                    break
                possible_date = now_date + relativedelta(months=j)
                possible_date = possible_date.strftime('%Y-%m')
                possible_contract_dates.append(possible_date)
            
            after_months = len(possible_contract_dates)
            while after_months < self.predict_range:
                next_date = now_date + relativedelta(months=after_months)
                if next_date.month in self.fixed_months:
                    possible_date = next_date.strftime('%Y-%m')
                    possible_contract_dates.append(possible_date)
                after_months += 1
            possible_contract['Possible Contract Date'][i] = possible_contract_dates
        
        trading_dates = []   #[1,1,1]
        delivery_months = [] #[1,2,3]
        for i in range(len(possible_contract)):
            contracts_num = len(possible_contract['Possible Contract Date'][i])
            for _ in range(contracts_num):
                trading_dates.append(possible_contract['Date'][i])
            if i == 0:
                delivery_months = possible_contract['Possible Contract Date'][i]
            else:
                delivery_months += possible_contract['Possible Contract Date'][i]
        contracts = pd.DataFrame(
            list(zip(trading_dates, delivery_months)),
            columns=['Trading Date', 'Contract Delivery Month']
        )
        
        '''
        Contract Delivery Month should not greater than end_date.
        Because we do not have data on the dates which are greater than end_date.
        '''
        contracts.drop(
            contracts[contracts['Contract Delivery Month']>self.end_date].index,
            inplace=True
        )
        contracts.reset_index(drop=True, inplace=True)
        return contracts
    
    def generate_contract(self, contract_name):
        contracts = self.calculate_contract_dates()
        target_df = None
        contracts_open = []
        if contract_name == 'MTX':
            target_df = pd.read_csv(f'./contracts/{contract_name}.csv')
            for i in range(len(contracts)):
                df = target_df[
                    (target_df['Trading Date'] >= contracts['Trading Date'][i])
                    & (target_df['Delivery Month'] == contracts['Contract Delivery Month'][i])
                ].sort_values(by=['Trading Date'])
                df = df.reset_index(drop=True)
                contracts_open.append(df['Open'][0])
        else:
            target_df = pd.read_csv(f'./data/daily/{contract_name}.csv')
            for i in range(len(contracts)):
                df = target_df[
                    (target_df['Date'] >= contracts['Trading Date'][i])
                ].sort_values(by=['Date'])
                df = df.reset_index(drop=True)
                contracts_open.append(df['Open'][0])

        contracts['Open'] = contracts_open
        return contracts
