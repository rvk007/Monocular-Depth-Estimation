import torch
import torch.nn as nn
import torch.nn.functional as F

from deepnet.model.learner import Model

class MaskNet3(nn.Module):
    def __init__(self):
        """Creates Masknet-3"""
        super(MaskNet3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer3= nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        

    def forward(self, x):
        bg = x['bg']
        bg_fg = x['bg_fg']
        bg = self.layer1(bg)
        bg_fg = self.layer2(bg_fg)

        out = torch.cat([bg, bg_fg], dim=1)
        out = self.layer3(out)

        return out

    def learner(self, model, tensorboard, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks):
        """Trains the model
        Arguments:
            model: Model to trained and validated
            tensorboard: Tensorboard instance for visualization
            dataset_train: Dataset training instance
            train_loader: Dataloader containing train data on the GPU/ CPU
            test_loader: Dataloader containing test data on the GPU/ CPU 
            device: Device on which model will be trained (GPU/CPU)
            optimizer: optimizer for the model
            criterion: Loss function
            epochs: Number of epochs to train the model
            metrics(bool): If metrics is to be displayed or not
                (default: False)
            callbacks: Scheduler to be applied on the model
                    (default : None)
        """

        learn = Model(model, tensorboard, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks)
        self.result = learn.fit()

    @property
    def results(self):
        """Returns model results"""
        return self.result