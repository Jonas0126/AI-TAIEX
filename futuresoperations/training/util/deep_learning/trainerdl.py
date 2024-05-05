from futuresoperations.training.util.prototype import TrainerPrototype
from futuresoperations.training.util.deep_learning import LSTM, TCN, CRNN
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import time
import copy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class TrainerDL(TrainerPrototype):
    def __init__(self, model_type, hyperparams_range):
        super().__init__(model_type, hyperparams_range)
        self.loss_history = None
        self.finetune_loss_history = None
    
    def create_models(self, input_dim, output_dim):
        models = []
        if self.model_type == 'LSTM':
            if 'hidden_dim' not in self.hyperparams_range:
                self.hyperparams_range['hidden_dim'] = [200]
            if 'n_layers' not in self.hyperparams_range:
                self.hyperparams_range['n_layers'] = [2]
            if 'dropout' not in self.hyperparams_range:
                self.hyperparams_range['dropout'] = [0.3]
            for hidden_dim in self.hyperparams_range['hidden_dim']:
                for n_layers in self.hyperparams_range['n_layers']:
                    for dropout in self.hyperparams_range['dropout']:
                        model = LSTM(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=n_layers,
                            dropout=dropout
                        )
                        hparams = {
                            'hidden_dim': hidden_dim,
                            'n_layers': n_layers,
                            'dropout': dropout
                        }
                        models.append((model, hparams))
            return models
        elif self.model_type == 'TCN':
            if 'n_layers' not in self.hyperparams_range:
                self.hyperparams_range['n_layers'] = [4]
            if 'kernel_size' not in self.hyperparams_range:
                self.hyperparams_range['kernel_size'] = [2]
            if 'dropout' not in self.hyperparams_range:
                self.hyperparams_range['dropout'] = [0.3]
            for n_layers in self.hyperparams_range['n_layers']:
                num_channels = [22] * n_layers
                for kernel_size in self.hyperparams_range['kernel_size']:
                    for dropout in self.hyperparams_range['dropout']:
                        model = TCN(
                            input_size=input_dim,
                            output_size=output_dim,
                            num_channels=num_channels,
                            kernel_size=kernel_size,
                            dropout=dropout
                        )
                        hparams = {
                            'n_layers': n_layers,
                            'kernel_size': kernel_size,
                            'dropout': dropout
                        }
                        models.append((model, hparams))
            return models
        elif self.model_type == 'CRNN':
            feature_size = input_dim * 2
            if 'hidden_dim' not in self.hyperparams_range:
                self.hyperparams_range['hidden_dim'] = [200]
            if 'n_layers' not in self.hyperparams_range:
                self.hyperparams_range['n_layers'] = [2]
            if 'dropout' not in self.hyperparams_range:
                self.hyperparams_range['dropout'] = [0.3]
            for hidden_dim in self.hyperparams_range['hidden_dim']:
                for n_layers in self.hyperparams_range['n_layers']:
                    for dropout in self.hyperparams_range['dropout']:
                        model = CRNN(
                            input_size=input_dim,
                            feature_size=feature_size,
                            hidden_size=hidden_dim,
                            output_size=output_dim,
                            n_layers=n_layers,
                            dropout=dropout
                        )
                        hparams = {
                            'hidden_dim': hidden_dim,
                            'n_layers': n_layers,
                            'dropout': dropout
                        }
                        models.append((model, hparams))
            return models

    def train_model(self, model, device, dataloaders,
                    criterion, optimizer, num_epochs=25):
        # since = time.time()
        trigger_times = 0
        patience = 10
        # val_acc_history = []
        loss_history = {'train':[], 'val':[]}

        best_model_wts = copy.deepcopy(model.state_dict())
        # best_acc = 0.0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            ##### learning rate scheduling #####
            if trigger_times < 2:
                lr = 0.0005
            elif trigger_times == 4:
                lr = 0.0001
            elif trigger_times == 6:
                lr = 0.00005
            elif trigger_times == 8:
                lr = 0.00001
            else:
                lr = 0.000005
            optimizer.param_groups[0]['lr'] = lr

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                # running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.float)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # _, preds = torch.max(outputs, 1) this is for classification

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    # running_corrects += torch.sum(preds == labels.data) this is for classification

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) this is for classification

                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) this is for classification
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())
                # if phase == 'val':
                #     val_acc_history.append(epoch_acc)
                loss_history[phase].append(epoch_loss)
                if phase == 'val' and epoch_loss < best_loss:
                    trigger_times = 0
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif phase == 'val' and epoch_loss >= best_loss:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print('Early stopping!\nEpoch:{}'.format(epoch))
                        model.load_state_dict(best_model_wts)
                        return model, loss_history
        # time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))
        # print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        print("Finished all epoch.")
        model.load_state_dict(best_model_wts)
        # val_history = {'acc': val_acc_history, 'loss': val_loss_history}
        return model, loss_history

    def test_model(self, model, device, testloader):
        model.eval()
        prediction = []
        mean_pred = []
        mean_label = []
        for inputs, labels in testloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.detach().cpu().numpy().squeeze()
                labels = labels.detach().cpu().numpy().squeeze()
                outputs = outputs*10000
                labels = labels*10000
                prediction.append(int(np.mean(outputs)))
                mean_pred.append(np.mean(outputs))
                mean_label.append(np.mean(labels))

        mean_pred = np.array(mean_pred)
        mean_label = np.array(mean_label)
        rmse = math.sqrt(mean_squared_error(mean_label,mean_pred))
        var = np.var(abs(mean_label-mean_pred))
        std = np.sqrt(np.var(abs(mean_label-mean_pred)))

        return prediction, rmse, var, std

    def fit(self, traindataset, testdataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_set_size = int(len(traindataset) * 0.9)
        valid_set_size = len(traindataset) - train_set_size
        train, validation = random_split(traindataset, [train_set_size, valid_set_size])
        testloader = DataLoader(
            dataset=testdataset,
            shuffle=False,
        )

        if 'batch_size' not in self.hyperparams_range:
            self.hyperparams_range['batch_size'] = [16,32,64]
        if 'epoch' not in self.hyperparams_range:
            self.hyperparams_range['epoch'] = [100]
        if 'lr' not in self.hyperparams_range:
            self.hyperparams_range['lr'] = [0.0005]

        for batch_size in self.hyperparams_range['batch_size']:
            trainloader = DataLoader(
                dataset=train,
                batch_size=batch_size,
                shuffle=True,
                # num_workers=4
            )
            validationloader = DataLoader(
                dataset=validation,
                batch_size=batch_size,
                shuffle=False,
                # num_workers=4
            )
            dataloaders = {'train':trainloader, 'val':validationloader}
            x, y = next(iter(trainloader))
            input_dim = x.size(2)
            output_dim = 22
            models = self.create_models(input_dim, output_dim)
            for epoch in self.hyperparams_range['epoch']:
                for lr in self.hyperparams_range['lr']:
                    for model, hparams in models:
                        model = model.to(device)
                        criterion = torch.nn.MSELoss(reduction='mean')
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=lr,
                            weight_decay=1e-8
                        )
                        best_model, loss_history = self.train_model(
                            model=model,
                            device=device,
                            dataloaders=dataloaders,
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=epoch
                        )
                        prediction, rmse, var, std = self.test_model(
                            model=best_model,
                            device=device,
                            testloader=testloader
                        )
                        if self.best_rmse > rmse:
                            self.best_rmse = rmse
                            self.best_model = best_model
                            self.best_hyperparams = hparams
                            self.best_hyperparams['batch_size'] = batch_size
                            self.best_hyperparams['epoch'] = epoch
                            self.best_hyperparams['lr'] = lr
                            self.loss_history = loss_history

    def finetuning(self, testdataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_set_size = int(len(testdataset) * 0.9)
        valid_set_size = len(testdataset) - train_set_size
        train, validation = random_split(testdataset, [train_set_size, valid_set_size])
        trainloader = DataLoader(
            dataset=train,
            batch_size=self.best_hyperparams['batch_size'],
            shuffle=True,
            # num_workers=4
        )
        validationloader = DataLoader(
            dataset=validation,
            batch_size=self.best_hyperparams['batch_size'],
            shuffle=False,
            # num_workers=4
        )
        dataloaders = {'train':trainloader, 'val':validationloader}
        model = self.best_model.to(device)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.best_hyperparams['lr'],
            weight_decay=1e-8
        )
        best_model, loss_history = self.train_model(
            model=model,
            device=device,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=self.best_hyperparams['epoch']
        )
        self.best_model = best_model
        self.finetune_loss_history = loss_history

    def draw_finetune_loss(self, model_path, predict_period):
        sns.set_style("darkgrid")
        fig = plt.figure()
        ax = sns.lineplot(
            data=np.array(self.finetune_loss_history['train']),
            label="Training",
            color='royalblue'
        )
        ax = sns.lineplot(
            data=np.array(self.finetune_loss_history['val']),
            label="Validation",
            color='tomato'
        )
        ax.set_xlabel("Epoch", size=14)
        ax.set_ylabel("Loss", size=14)
        ax.set_title("Epoch Loss", size=14, fontweight='bold')
        fig.set_figheight(6)
        fig.set_figwidth(16)

        plt.savefig(f'{model_path}/predmonth_{predict_period}_finetune.png')

        f = open(f'{model_path}/predmonth_{predict_period}_finetune.txt', 'w')
        f.write(f'Train: {self.loss_history["train"]}')
        f.write(f'Val: {self.loss_history["val"]}')

    def draw(self, model_path, predict_period):
        sns.set_style("darkgrid")
        fig = plt.figure()
        ax = sns.lineplot(
            data=np.array(self.loss_history['train']),
            label="Training",
            color='royalblue'
        )
        ax = sns.lineplot(
            data=np.array(self.loss_history['val']),
            label="Validation",
            color='tomato'
        )
        ax.set_xlabel("Epoch", size=14)
        ax.set_ylabel("Loss", size=14)
        ax.set_title("Epoch Loss", size=14, fontweight='bold')
        fig.set_figheight(6)
        fig.set_figwidth(16)

        plt.savefig(f'{model_path}/predmonth_{predict_period}.png')

        f = open(f'{model_path}/predmonth_{predict_period}.txt', 'w')
        f.write(f'Train: {self.loss_history["train"]}')
        f.write(f'Val: {self.loss_history["val"]}')

        self.draw_finetune_loss(model_path, predict_period)