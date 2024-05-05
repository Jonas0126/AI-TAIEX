from os import path
from futuresoperations.training.util.prototype import TrainerPrototype
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import numpy as np

class TrainerML(TrainerPrototype):
    def __init__(self, model_type, hyperparams_range):
        super().__init__(model_type, hyperparams_range)
    
    def create_models(self):
        models = []
        if self.model_type == 'LR':
            models.append((LinearRegression(), {}))
            return models
        elif self.model_type == 'XGB':
            if 'objective' not in self.hyperparams_range:
                self.hyperparams_range['objective'] = ['reg:squarederror']
            if 'seed' not in self.hyperparams_range:
                self.hyperparams_range['seed'] = [123]
            for objective in self.hyperparams_range['objective']:
                for seed in self.hyperparams_range['seed']:
                    model = xgb.XGBRegressor(
                        objective=objective,
                        seed=seed
                    )
                    hparams = {
                        'objective': objective,
                        'seed': seed
                    }
                    models.append((model, hparams))
            return models

    def fit(self, traindataset, testdataset):
        x_train, y_train = traindataset
        x_test, y_test = testdataset
        start, end = self.hyperparams_range['month_length']
        for month_length in range(start, end+1):
            # take last 'month_length' data(eg. 24 = take recent 2 years data)
            X = x_train[-month_length:]
            y = y_train[-month_length:]
            models = self.create_models()
            for model, hparams in models:
                model_fit = model.fit(X, y)
                rmse = (y_test-model_fit.predict(x_test)) ** 2
                rmse = np.mean(rmse) ** (0.5)
                if rmse < self.best_rmse:
                    self.best_model = model_fit
                    self.best_rmse = rmse
                    self.best_hyperparams = hparams
                    self.best_hyperparams['month_length'] = month_length
    
    def finetuning(self, testdataset):
        pass

    def draw(self, model_path, predict_period):
        pass