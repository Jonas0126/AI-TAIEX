from futuresoperations.training.util.prototype import TrainerPrototype
from futuresoperations.training.util.machine_learning import TrainerML
from futuresoperations.training.util.deep_learning import TrainerDL


class Trainer(TrainerPrototype):
    '''Trainer is to fine tune the model
    '''
    def __init__(self, model_type, hyperparams_range):
        super().__init__(model_type, hyperparams_range)
        self.trainer = self.select_trainer(model_type, hyperparams_range)

    def select_trainer(self, model_type, hyperparams_range):
        if model_type == 'LR' or model_type == 'XGB':
            if 'month_length' not in hyperparams_range:
                hyperparams_range['month_length'] = (24, 72)
            return TrainerML(model_type, hyperparams_range)
        else:
            return TrainerDL(model_type, hyperparams_range)
    
    def fit(self, traindataset, testdataset):
        self.trainer.fit(traindataset, testdataset)
        return self.trainer
    
    def finetuning(self, testdataset):
        self.trainer.finetuning(testdataset)
    
    def draw(self, model_path, predict_period):
        self.trainer.draw(model_path, predict_period)