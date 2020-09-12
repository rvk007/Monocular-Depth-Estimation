import torch
import torch.nn as nn
from torch.nn import functional as F

from deepnet.model.losses.dice_loss import DiceLoss
from deepnet.model.losses.loss import mse_loss, rmse_loss, bce_loss, bcewithlogits_loss
from deepnet.model.losses.ssim import SSIM_Loss


class BCE_RMSE_LOSS(nn.Module):
    def __init__(self):
        super(BCE_RMSE_LOSS, self).__init__()
        self.bce_loss = bcewithlogits_loss()
        self.rmse_loss = rmse_loss()

    def forward(self,prediction, label):
        loss = self.bce_loss(prediction['bg_fg_mask'], label['bg_fg_mask']) + (2 * self.rmse_loss(prediction['bg_fg_depth'], label['bg_fg_depth']))

        return loss


class SSIM_RMSE_LOSS(nn.Module):
    def __init__(self):
        super(SSIM_RMSE_LOSS, self).__init__()
        self.ssim_loss = SSIM_Loss()
        self.rmse_loss = rmse_loss()

    def forward(self,prediction, label):
        loss = self.ssim_loss(prediction['bg_fg_mask'], label['bg_fg_mask']) + (2 * self.rmse_loss(prediction['bg_fg_depth'], label['bg_fg_depth']))

        return loss


class BCE_SSIM_LOSS(nn.Module):
    def __init__(self):
        super(BCE_SSIM_LOSS, self).__init__()
        self.bce_loss = bcewithlogits_loss()
        self.ssim_loss = SSIM_Loss()

    def forward(self,prediction, label):
        loss = self.bce_loss(prediction['bg_fg_mask'], label['bg_fg_mask']) + (2 * self.ssim_loss(prediction['bg_fg_depth'], label['bg_fg_depth']))

        return loss


class RMSE_SSIM_LOSS(nn.Module):
    def __init__(self):
        super(RMSE_SSIM_LOSS, self).__init__()
        self.bce_loss = rmse_loss()
        self.ssim_loss = SSIM_Loss()

    def forward(self,prediction, label):
        loss = self.rmse_loss(prediction['bg_fg_mask'], label['bg_fg_mask']) + (2 * self.ssim_loss(prediction['bg_fg_depth'], label['bg_fg_depth']))

        return loss

class SSIM_DICE_LOSS(nn.Module):
    def __init__(self):
        super(SSIM_DICE_LOSS, self).__init__()
        self.ssim_loss = SSIM_Loss()
        self.dice_loss = DiceLoss()

    def forward(self,prediction, label):
        loss = self.ssim_loss(prediction['bg_fg_mask'], label['bg_fg_mask']) + (2 * self.dice_loss(prediction['bg_fg_depth'], label['bg_fg_depth']))

        return loss

class RMSE_DICE_LOSS(nn.Module):
    def __init__(self):
        super(RMSE_DICE_LOSS, self).__init__()
        self.rmse_loss = rmse_loss()
        self.dice_loss = DiceLoss()

    def forward(self,prediction, label):
        loss = self.rmse_loss(prediction['bg_fg_mask'], label['bg_fg_mask']) + (2 * self.dice_loss(prediction['bg_fg_depth'], label['bg_fg_depth']))

        return loss

class BCEDiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        """BCEDice Loss.
        Args:
            smooth (float, optional): Smoothing value.
        """
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, input, target):
        """Calculate BCEDice Loss.
        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
        
        Returns:
            BCEDice loss
        """
        
        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        dice_loss = self.dice(torch.sigmoid(input), target)
        return bce_loss + 2 * dice_loss

class RmseBceDiceLoss(nn.Module):
    
    def __init__(self):
        super(RmseBceDiceLoss, self).__init__()
        self.rmse = rmse_loss()
        self.bce_dice = BCEDiceLoss()

    def forward(self, prediction, label):
        return (
            2 * self.rmse(prediction['bg_fg_mask'], label['bg_fg_mask']) +
            self.bce_dice(prediction['bg_fg_depth'], label['bg_fg_depth'])
        )