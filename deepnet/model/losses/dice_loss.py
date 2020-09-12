# # This code is referenced from https://github.com/milesial/Pytorch-UNet/blob/master/dice_loss.py

# import torch
# from torch.autograd import Function


# class DiceLoss(Function):
#     """Dice coeff for individual examples"""

#     def forward(self, input, target):
#         self.save_for_backward(input, target)
#         eps = 0.0001
#         self.inter = torch.dot(input.view(-1), target.view(-1))
#         self.union = torch.sum(input) + torch.sum(target) + eps

#         t = 1- ((2 * self.inter.float() + eps) / self.union.float())
#         return t

#     # This function has only a single output, so it gets only one gradient
#     def backward(self, grad_output):

#         input, target = self.saved_variables
#         grad_input = grad_target = None

#         if self.needs_input_grad[0]:
#             grad_input = grad_output * 2 * (target * self.union - self.inter) \
#                          / (self.union * self.union)
#         if self.needs_input_grad[1]:
#             grad_target = None

#         return grad_input, grad_target


#     def __call__(self, input, target):
#         """Dice coeff for batches"""
#         if input.is_cuda:
#             s = torch.FloatTensor(1).cuda().zero_()
#         else:
#             s = torch.FloatTensor(1).zero_()

#         for i, c in enumerate(zip(input, target)):
#             s = s + DiceLoss().forward(c[0], c[1])

#         return s / (i + 1)


from torch import nn
from torch.nn import functional as F
import torch


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        """Dice Loss.
        Args:
            smooth (float, optional): Smoothing value. A larger
                smooth value (also known as Laplace smooth, or
                Additive smooth) can be used to avoid overfitting.
                (default: 1)
        """
        super(DiceLoss, self).__init__()

        self.smooth = 1

    def forward(self, input, target):
        """Calculate Dice Loss.
        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
        
        Returns:
            dice loss
        """
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()

        return 1 - ((2. * intersection + self.smooth) / (union + self.smooth))

