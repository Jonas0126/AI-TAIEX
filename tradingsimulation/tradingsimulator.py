from tradingsimulation.contract import ContractGenerator, ContractPreprocessor
from tradingsimulation.trade import Trader
import os

class TradingSimulator:
    '''
    Futures rule:
        1. fixed_months: 每年固定一定有幾月的期貨
        2. total_front_month: 
    Trading type:
        1. one trade each month, only long
        2. one trade each month, only short
        3. one trade each month, long or short
        TODO:
            4. two trades each month, only long
            5. two trades each month, only short
            6. two trades each month, long or short
            7. two trades each month, one long one short
    '''
    def __init__(self, start_date, end_date, target_name, contract_name,
                futures_rule, fp=None, finetune_type=1, emsembler=None):
        self.start_date = start_date
        self.end_date = end_date
        self.target_name = target_name
        self.contract_name = contract_name
        self.futures_rule = futures_rule
        self.fp = fp
        self.finetune_type = finetune_type
        self.emsembler = emsembler
        self.cg = None
        self.cp = None
        self.trader = None
        self.trading_path = ''
        self.__create_trading_path()
    
    def __create_trading_path(self):
        trade_time_folder = f'{self.start_date}_{self.end_date}'
        results_folder = 'trading_results'
        if self.finetune_type == 0:
            results_folder += '_no_finetune'
        elif self.finetune_type == 2:
            results_folder += '_finetune_once'
        elif self.finetune_type == 3:
            results_folder += '_finetune_constant'
        if self.emsembler is None:
            self.trading_path = f'{self.fp.model_path}/{results_folder}/{trade_time_folder}'
        else:
            self.trading_path = f'{self.emsembler.model_path}/{results_folder}/{trade_time_folder}'
        if not os.path.exists(self.trading_path):
            os.makedirs(self.trading_path)
        
    def generate_contract(self):
        fixed_months = self.futures_rule['fixed months']
        total_front_month = self.futures_rule['total front month']
        if self.fp is None:
            self.cg = ContractGenerator(
                start_date=self.start_date,
                end_date=self.end_date,
                fixed_months=fixed_months,
                total_front_month=total_front_month,
                predict_range=self.emsembler.predict_range
            )
        else:
            self.cg = ContractGenerator(
                start_date=self.start_date,
                end_date=self.end_date,
                fixed_months=fixed_months,
                total_front_month=total_front_month,
                predict_range=self.fp.predict_range
            )
        contracts = self.cg.generate_contract(self.contract_name)
        return contracts
    
    def preprocessing(self, contracts, trading_type):
        self.cp = ContractPreprocessor(
            start_date=self.start_date,
            end_date=self.end_date,
            contracts=contracts,
            fp=self.fp,
            finetune_type=self.finetune_type,
            emsembler=self.emsembler
        )
        self.cp.load_predict()
        self.cp.select_contract(trading_type)
        return self.cp.contracts
    
    def trading(self, contracts, stop_loss, tolerance):
        delivery_week = self.futures_rule['delivery week']
        delivery_weekday = self.futures_rule['delivery weekday']
        self.trader = Trader(
            contract_name=self.contract_name,
            target_name=self.target_name,
            contracts=contracts,
            delivery_week=delivery_week,
            delivery_weekday=delivery_weekday,
            stop_loss=stop_loss,
            tolerance=tolerance
        )
        contracts = self.trader.trading()
        return contracts
    
    def simulation(self, trading_type, stop_loss, tolerance=100):
        '''
        fp: futures predictor
        trading_type: 
            1. long
            2. short
            3. longorshort
        stop_loss: True or False
        tolerance: number
        '''
        contracts = self.generate_contract()
        contracts = self.preprocessing(contracts, trading_type)
        contracts = self.trading(contracts, stop_loss, tolerance)
        if stop_loss:
            contract_path = f'{self.trading_path}/stop_loss'
            if not os.path.exists(contract_path):
                os.makedirs(contract_path)
            contract_name = f'{trading_type}_contracts_tolerance_{tolerance}.csv'
            contracts.to_csv(f'{contract_path}/{contract_name}', index=False)
        else:
            contracts.to_csv(f'{self.trading_path}/{trading_type}_contracts.csv', index=False)
        return contracts