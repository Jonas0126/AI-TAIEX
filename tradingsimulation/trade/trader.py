import pandas as pd
from datetime import datetime, date, timedelta

class Trader:
    def __init__(self, contract_name, target_name, contracts, delivery_week=3,
                 delivery_weekday=3, stop_loss=True, tolerance=100):
        self.contract_name = contract_name
        self.target_name = target_name
        self.contracts = contracts
        self.delivery_week = delivery_week
        self.delivery_weekday = delivery_weekday
        self.stop_loss = stop_loss
        self.tolerance = tolerance
    
    def load_daily_data(self):
        if self.contract_name != 'MTX':
            # first method
            df = pd.read_csv(f'./data/daily/{self.target_name}.csv')
        else:
            # second method
            df = pd.read_csv(f'./contracts/{self.contract_name}.csv')
        return df
    
    def check_delivery_day(self, curr_date, possible_delivery_date):
        possible_delivery_date = datetime.strptime(possible_delivery_date, '%Y-%m-%d')
        curr_date = datetime.strptime(curr_date, '%Y-%m-%d')
        delivery_month = possible_delivery_date.month
        if delivery_month != curr_date.month:
            return False
        delivery_month_first_day = possible_delivery_date.replace(day=1)
        delivery_week = delivery_month_first_day + timedelta(weeks=self.delivery_week-1)
        delivery_week = delivery_week.isocalendar()[1]
        curr_week = curr_date.isocalendar()[1]
        curr_weekday = curr_date.isoweekday()
        if ((curr_week==delivery_week) and
            (curr_weekday==self.delivery_weekday)) or curr_week > delivery_week:
            return True
        else:
            return False
    
    def get_stop_loss_point(self, target_contract):
        std = 100
        if target_contract['Trading Type'] == 'Short':
            price1 = target_contract['Predict'] + std
            price2 = target_contract['Open'] + self.tolerance
            return max(price1, price2)
        else:
            price1 = target_contract['Predict'] - std
            price2 = target_contract['Open'] - self.tolerance
            return min(price1, price2)
    
    def sell_contract(self, df_daily, target_contract):
        start_date = target_contract['Trading Date']
        delivery_month = target_contract['Contract Delivery Month']
        end_date = pd.to_datetime(delivery_month)
        end_date = str(date(end_date.year, end_date.month, 28))

        # first method
        if self.contract_name != 'MTX':
            df_target_daily = df_daily[
                (df_daily['Date']>=start_date)
                & (df_daily['Date']<=end_date)
            ].reset_index(drop=True)
        else:
            # second method
            # print('##########################')
            # print('start date', start_date)
            df_target_daily = df_daily[
                (df_daily['Trading Date']>=start_date)
                & (df_daily['Trading Date']<=end_date)
                & (df_daily['Delivery Month']==delivery_month)
            ].reset_index(drop=True).sort_values(by=['Trading Date'])
            df_target_daily.rename(columns={'Trading Date': 'Date'}, inplace=True)
            # print(df_target_daily)
            # print('3333333333333333333333333333333333333333')

        is_stop_loss = False
        pred_value = target_contract['Predict']
        for i in range(len(df_target_daily)):
            if target_contract['Trading Type'] == 'Short':
                # we always consider the worst condition
                if self.stop_loss:
                    stop_loss_point = self.get_stop_loss_point(target_contract)
                    if df_target_daily['High'][i] >= stop_loss_point:
                        is_stop_loss = True
                        return df_target_daily['Date'][i], stop_loss_point, is_stop_loss
                if df_target_daily['Low'][i] <= pred_value:
                    return df_target_daily['Date'][i], pred_value, is_stop_loss
            else:
                # we always consider the worst condition
                if self.stop_loss:
                    stop_loss_point = self.get_stop_loss_point(target_contract)
                    if df_target_daily['Low'][i] <= stop_loss_point:
                        is_stop_loss = True
                        return df_target_daily['Date'][i], stop_loss_point, is_stop_loss
                if df_target_daily['High'][i] >= pred_value:
                    return df_target_daily['Date'][i], pred_value, is_stop_loss
            if self.check_delivery_day(df_target_daily['Date'][i], end_date):
                return df_target_daily['Date'][i], df_target_daily['Close'][i], is_stop_loss
                
    def trading(self):
        df_daily = self.load_daily_data()
        stop_loss_points = []
        is_stop_loss_list = []
        selling_dates = []
        selling_points = []
        profits = []
        for i in range(len(self.contracts)):
            if self.contracts['Selection'][i] == 'Y':
                selling_date, selling_point, is_stop_loss = self.sell_contract(
                    df_daily=df_daily,
                    target_contract=self.contracts.iloc[i]
                )
                if self.stop_loss:
                    stop_loss_point = self.get_stop_loss_point(self.contracts.iloc[i])
                    stop_loss_points.append(stop_loss_point)
                else:
                    stop_loss_points.append('')
                is_stop_loss_list.append(is_stop_loss)
                selling_dates.append(selling_date)
                selling_points.append(selling_point)
                profit = selling_point - self.contracts['Open'][i]
                if self.contracts['Trading Type'][i] == 'Short':
                    profit = -profit
                profits.append(profit)
            else:
                stop_loss_points.append('')
                is_stop_loss_list.append('')
                selling_dates.append('')
                selling_points.append('')
                profits.append(0)
        
        self.contracts['Stop Loss Point'] = stop_loss_points
        self.contracts['Is Stop Loss'] = is_stop_loss_list
        self.contracts['Selling Date'] = selling_dates
        self.contracts['Selling Point'] = selling_points
        self.contracts['Profit'] = profits
        total = self.contracts["Profit"].sum()
        total_list = [""]*len(self.contracts)
        total_list[0] = total
        self.contracts["Total"] = total_list
        return self.contracts