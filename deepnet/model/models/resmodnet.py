import torch
import torch.nn as nn
import torch.nn.functional as F

from deepnet.model.learner import Model

class ModBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Creates the basic block of RESNET-18
        
        Arguments:
            in_planes : Number of input channels
            planes : Number of output channels
            stride : Value of stride in the model (By default = 1)
        """
        super(ModBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResModNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        """
        Creates RESNET-18
        Arguments:
            block : Basic block of resnet
            num_blocks : List of number of convolutions in each block
            num_classes : Number of labels in dataset
        """
        super(ResModNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Arguments:
            block : The basic block for the coresponding layer
            planes : Number of output channels
            num_blocks : Number of convolutions for this block
            stride : Value of stride
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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