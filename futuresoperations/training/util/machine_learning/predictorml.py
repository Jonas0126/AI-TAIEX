from futuresoperations.training.util.prototype import PredictorPrototype
import numpy as np

class PredictorML(PredictorPrototype):
    def __init__(self, model):
        super().__init__(model)
    
    def inference(self, dataset):
        X, y = dataset
        prediction = self.model.predict(X)
        rmse = (y-prediction) ** 2
        rmse = np.mean(rmse) ** (0.5)
        return prediction, rmse