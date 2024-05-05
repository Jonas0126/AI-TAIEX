import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

class ContractPreprocessor:
    def __init__(self, start_date, end_date, contracts, fp, finetune_type, emsembler):
        self.start_date = start_date
        self.end_date = end_date
        self.contracts = contracts
        self.fp = fp
        self.finetune_type = finetune_type
        self.emsembler = emsembler
    
    def load_predict(self):
        # here is a little tricky
        # the last month of the start date is the predict start date
        # because predict_from_date is predict from features date
        if self.emsembler is None:
            pred_df = self.fp.inference_by_load(
                sdate=self.start_date,
                edate=self.end_date,
                finetune_type=self.finetune_type
            )
            
            pred_list = []
            features_date = pd.to_datetime(pred_df['Features Date'], format='%Y-%m')
            pred_df['Trading Date'] = features_date + pd.DateOffset(months=1)
            for i in range(len(self.contracts)):
                pred_value = pred_df[
                    (pred_df['Trading Date']==self.contracts['Trading Date'][i])
                    & (pred_df['Predict Date']==self.contracts['Contract Delivery Month'][i])
                ]
                pred_value = pred_value['Predict'].values[0]
                pred_list.append(pred_value)
            self.contracts['Predict'] = pred_list
        else:
            pred_df1 = self.emsembler.fp1.inference_by_load(
                sdate=self.start_date,
                edate=self.end_date
            )
            pred_df2 = self.emsembler.fp2.inference_by_load(
                sdate=self.start_date,
                edate=self.end_date
            )
            
            pred_list = []
            features_date = pd.to_datetime(pred_df1['Features Date'], format='%Y-%m')
            pred_df1['Trading Date'] = features_date + pd.DateOffset(months=1)
            pred_df2['Trading Date'] = features_date + pd.DateOffset(months=1)
            for i in range(len(self.contracts)):
                pred_value1 = pred_df1[
                    (pred_df1['Trading Date']==self.contracts['Trading Date'][i])
                    & (pred_df1['Predict Date']==self.contracts['Contract Delivery Month'][i])
                ]
                pred_value2 = pred_df2[
                    (pred_df2['Trading Date']==self.contracts['Trading Date'][i])
                    & (pred_df2['Predict Date']==self.contracts['Contract Delivery Month'][i])
                ]
                pred_value1 = pred_value1['Predict'].values[0]
                pred_value2 = pred_value2['Predict'].values[0]
                pred_list.append((pred_value1+pred_value2)/2)
            self.contracts['Predict'] = pred_list
    
    def select_contract(self, trading_type):
        selection = []
        trading_types = []
        self.contracts['Difference'] = self.contracts['Predict'] - self.contracts['Open']
        for i in range(len(self.contracts)):
            max_pos_diff = self.contracts[
                self.contracts['Trading Date']==self.contracts['Trading Date'][i]
            ]['Difference'].max()
            min_neg_diff = self.contracts[
                self.contracts['Trading Date']==self.contracts['Trading Date'][i]
            ]['Difference'].min()
            
            if trading_type == 'long':
                if (self.contracts['Difference'][i]==max_pos_diff) and (max_pos_diff>0):
                    selection.append('Y')
                    trading_types.append('Long')
                else:
                    selection.append('N')
                    trading_types.append('')
            elif trading_type == 'short':
                if (self.contracts['Difference'][i]==min_neg_diff) and (min_neg_diff<0):
                    selection.append('Y')
                    trading_types.append('Short')
                else:
                    selection.append('N')
                    trading_types.append('')
            else:
                target_diff = None
                if abs(max_pos_diff) > abs(min_neg_diff):
                    target_diff = max_pos_diff
                else:
                    target_diff = min_neg_diff
                if (self.contracts['Difference'][i]==target_diff) and (target_diff!=0):
                    selection.append('Y')
                    if target_diff > 0:
                        trading_types.append('Long')
                    else:
                        trading_types.append('Short')
                else:
                    selection.append('N')
                    trading_types.append('')
        
        self.contracts['Selection'] = selection
        self.contracts['Trading Type'] = trading_types