import torch.nn as nn
import math
import torch

def mse_loss(size_average=None, reduce=None, reduction='mean'):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input xx and target yy
    Arguments:
        size_average (bool, optional) : By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple
            elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False.
            (default: True)
        reduce (bool, optional) : By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is 
            False, returns a loss per batch element instead and ignores size_average.
            (default: True)
        reduction (string, optional) : Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            (default: 'mean')
    Returns:
        MSELoss
    """
    return nn.MSELoss(size_average, reduce, reduction)


def bce_loss(weight=None, size_average=None, reduce=None, reduction='mean'):
    """Creates a criterion that measures the Binary Cross Entropy
    between the target and the output
    Arguments:
        weights(Tensor, optional) :  A manual rescaling weight given to the loss of each batch element.
        size_average (bool, optional) : By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple
            elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False.
            (default: True)
        reduce (bool, optional) : By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is 
            False, returns a loss per batch element instead and ignores size_average.
            (default: True)
        reduction (string, optional) : Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            (default: 'mean')
    Returns:
        BCEloss
    """
    return nn.BCEloss(weight, size_average, reduce, reduction)


def bcewithlogits_loss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    """Creates a criterion that combines a `Sigmoid` layer and the `BCELoss` in one single
    class
    Arguments:
        weights(Tensor, optional) :  A manual rescaling weight given to the loss of each batch element.
        size_average (bool, optional) : By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple
            elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False.
            (default: True)
        reduce (bool, optional) : By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is 
            False, returns a loss per batch element instead and ignores size_average.
            (default: True)
        reduction (string, optional) : Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            (default: 'mean')
        pos_weight (Tensor, optional) : a weight of positive examples. Must be a vector with length equal to the number of classes.
    Returns:
        BCEWithLogitsLoss
    """
    return nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


def rmse_loss():
    return RMSELoss()


