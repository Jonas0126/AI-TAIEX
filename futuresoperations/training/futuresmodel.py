from futuresoperations.training.util import Trainer
from futuresoperations.training.util import Predictor

class FuturesModel:
    def __init__(self, train_dates, test_dates, target_name,
                 predict_period=1, model_type='LR', hyperparams=None):
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.target_name = target_name
        self.predict_period = predict_period
        self.model_type = model_type
        self.hyperparams = hyperparams
        self.best_model = None
        self.best_rmse = float('inf')
        self.best_hyperparams = None
        self.trainer = None

    def draw_loss(self, model_path):
        self.trainer.draw(model_path, self.predict_period)

    def pipelining(self, traindataset, testdataset):
        '''
        This pipeline includes training and testing model
        '''
        trainer = Trainer(
            model_type=self.model_type,
            hyperparams_range=self.hyperparams
        )
        self.trainer = trainer.fit(traindataset, testdataset)
        self.best_model = self.trainer.best_model
        self.best_rmse = self.trainer.best_rmse
        self.best_hyperparams = self.trainer.best_hyperparams
    
    def finetuning(self, testdataset):
        self.trainer.finetuning(testdataset)
        self.best_model = self.trainer.best_model

    def inference(self, dataset):
        predictor = Predictor(self.best_model, self.model_type)
        return predictor.inference(dataset)