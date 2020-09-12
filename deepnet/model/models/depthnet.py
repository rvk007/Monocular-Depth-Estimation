import torch
import torch.nn as nn
import torch.nn.functional as F

from deepnet.model.learner import Model

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.skip_conn = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 += self.skip_conn(x)
        x1 = F.relu(x1)
        return x1


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()

        self.layer = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, decode, encode, change=True):
        x = decode
        x = self.layer(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x += encode
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class DepthMaskNet8(nn.Module):
    def __init__(self):
        """Creates DepthMaskNet-8"""
        super(DepthMaskNet8, self).__init__()

        self.layer1a = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer1b = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.last_mask = nn.Conv2d(16, 1, kernel_size=1)
        self.last_depth = nn.Conv2d(16, 1, kernel_size=1)

        self.encodebg = Encoder(16, 16) #112
        self.encodebg_fg = Encoder(16, 16) #112
        
        #Common
        self.encode1 = Encoder(32,64) #56
        self.encode2 = Encoder(64,128) #28
        self.encode3 = Encoder(128,256) #14

        #Mask
        self.decode1_map = Decoder(256,128) #28
        self.decode2_map = Decoder(128,64) #56
        self.decode3_map = Decoder(64,32) #112
        self.decode4_map = Decoder(32,16) #224

        #Depth
        self.encode4 = Encoder(256,512) #7

        self.decode1_depth = Decoder(512,256) #14
        self.decode2_depth = Decoder(256,128) #28
        self.decode3_depth = Decoder(128,64) #56
        self.decode4_depth = Decoder(64,32) #112
        self.decode5_depth = Decoder(32,16) #224
        
    def forward(self, feature):
        bg = feature['bg']
        bg_fg = feature['bg_fg']
        bg = self.layer1a(bg)
        bg_fg = self.layer1b(bg_fg)

        x1 = self.encodebg(bg)
        x2 = self.encodebg_fg(bg_fg)

        x3 = torch.cat([x1, x2], dim=1)

        #Common
        x4 = self.encode1(x3)
        x5 = self.encode2(x4)
        x6 = self.encode3(x5)

        # Mask Prediction
        m1 = self.decode1_map(x6,x5)
        m2 = self.decode2_map(m1,x4)
        m3 = self.decode3_map(m2,x3)
        m4 = self.decode4_map(m3,bg_fg)
        Out_M = self.last_mask(m4)


        #Depth Prediction
        x7 = self.encode4(x6)

        d1 = self.decode1_depth(x7,x6)
        d2 = self.decode2_depth(d1,x5)
        d2 += m1
        d3 = self.decode3_depth(d2,x4)
        d3 += m2
        d4 = self.decode4_depth(d3,x3)
        d4 += x3
        d5 = self.decode5_depth(d4,bg_fg)
        Out_D = self.last_depth(d5)

        return Out_M, Out_D


    def learner(self, start_epoch, model, tensorboard, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks):
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

    def learner(self, start_epoch, model, tensorboard, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks):
        learn = Model(start_epoch, model, tensorboard, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks)
        self.result = learn.fit()

    @property
    def results(self):
        """Returns model results"""
        return self.result