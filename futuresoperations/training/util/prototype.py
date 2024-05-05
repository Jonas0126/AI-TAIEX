from abc import ABC, abstractmethod

class TrainerPrototype(ABC):
    def __init__(self, model_type, hyperparams_range):
        self.model_type = model_type
        self.hyperparams_range = hyperparams_range
        self.best_model = None
        self.best_rmse = float('inf')
        self.best_hyperparams = None

    @abstractmethod
    def fit(self, traindataset, testdataset):
        pass

    @abstractmethod
    def finetuning(self, testdataset):
        pass

    @abstractmethod
    def draw(self, model_path, predict_period):
        pass

class PredictorPrototype(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def inference(self, dataset):
        pass