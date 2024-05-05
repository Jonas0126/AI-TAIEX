from futuresoperations.training.util.prototype import PredictorPrototype
from futuresoperations.training.util.machine_learning import PredictorML
from futuresoperations.training.util.deep_learning import PredictorDL

class Predictor(PredictorPrototype):
    def __init__(self, model, model_type):
        super().__init__(model)
        self.predictor = self.select_predictor(model_type)
    
    def select_predictor(self, model_type):
        ml_model = ['LR', 'XGB']
        if model_type in ml_model:
            return PredictorML(self.model)
        else:
            return PredictorDL(self.model)

    def inference(self, dataset):
        return self.predictor.inference(dataset)