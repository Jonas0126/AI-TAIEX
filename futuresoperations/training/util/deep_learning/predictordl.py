from futuresoperations.training.util.prototype import PredictorPrototype
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
import math

class PredictorDL(PredictorPrototype):
    def __init__(self, model):
        super().__init__(model)
    
    def inference(self, dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
        )
        self.model.eval()
        prediction = []
        mean_pred = []
        mean_label = []
        for inputs, labels in dataloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            with torch.no_grad():
                outputs = self.model(inputs)
                outputs = outputs.detach().cpu().numpy().squeeze()
                labels = labels.detach().cpu().numpy().squeeze()
                outputs = outputs
                labels = labels
                prediction.append(np.mean(outputs))
                mean_pred.append(np.mean(outputs))
                mean_label.append(np.mean(labels))

        mean_pred = np.array(mean_pred)
        mean_label = np.array(mean_label)
        rmse = math.sqrt(mean_squared_error(mean_label,mean_pred))
        var = np.var(abs(mean_label-mean_pred))
        std = np.sqrt(np.var(abs(mean_label-mean_pred)))
        return prediction, rmse