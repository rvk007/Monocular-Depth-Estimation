import torch
import torch.nn as nn
import torch.nn.functional as F

from deepnet.model.models.resnet import BasicBlock
from deepnet.model.learner import Model

class CustomBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride, skip_connection):
        """Creates the basic block of CustomNet
        Arguments:
            in_planes : Number of input channels
            planes : Number of output channels
            stride : Value of stride in the model (By default = 1)
            skip_connection : True if skip connection to be applied, else False
        """
        super(CustomBlock, self).__init__()
        self.skip_connection = skip_connection
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        
        self.layer = self._make_layer(planes, stride=stride)

    def _make_layer(self, planes, stride):
        layers = []
        layers.append(BasicBlock(planes, planes, stride))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv(x)
        if self.skip_connection:
            x = x + self.layer(x)
        return x

class CustomNet(nn.Module):

    def __init__(self, block, num_classes=10):
        """Creates CustomNet
        Arguments:
            block : Basic block of resnet
            num_classes : Number of labels in dataset
        """
        super(CustomNet, self).__init__()
        self.in_planes = 64

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._custom_layer(block, 64, 128, stride=1, skip_connection=True)
        self.layer2 = self._custom_layer(block, 128, 256, stride=1, skip_connection=False)
        self.layer3 = self._custom_layer(block, 256, 512, stride=1, skip_connection=True)
        self.maxpool = nn.MaxPool2d(4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _custom_layer(self, block, in_planes, planes, stride, skip_connection):
        """Add layers to the model
        Arguments:
            block : The basic block for the coresponding layer
            in_planes : Number of input channels
            planes : Number of output channels
            stride : Value of stride
            skip_connection : True if skip connection to be applied, else False
        """
        layers = []
        layers.append(block(in_planes, planes, stride, skip_connection))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

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