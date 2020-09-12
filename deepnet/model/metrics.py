import torch
import math


class Metric:
    """
    Calculates the mean absolute error.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """
    def __init__(self):
        self._sum_of_errors_mae = 0.0
        self._sum_of_errors_rmse = 0.0
        self._sum_of_errors_mare = 0.0
        self._sum_of_errors_iou = 0.0
        self._sum_of_errors_rmse_new = 0.0

        self._num_examples_mae = 0
        self._num_examples_rmse = 0
        self._num_examples_mare = 0
        self._num_examples_iou = 0
        self._num_examples_rmse_new = 0

    def reset(self):
        self._sum_of_errors_mae = 0.0
        self._sum_of_errors_rmse = 0.0
        self._sum_of_errors_mare = 0.0
        self._sum_of_errors_iou = 0.0
        self._sum_of_errors_rmse_new = 0.0

        self._num_examples_mae = 0
        self._num_examples_rmse = 0
        self._num_examples_mare = 0
        self._num_examples_iou = 0
        self._num_examples_rmse_new = 0
    
    def mae(self, output):
        """Calculate cummulative metric value
        Arguments:
            output(tuple): A tuple (prediction tensor, ground truth tensor)
        """
        y_pred, y = output
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._sum_of_errors_mae += torch.sum(absolute_errors).item()
        self._num_examples_mae += y.shape[0]
        
        if self._num_examples_mae == 0:
            raise NotComputableError(f"MeanAbsoluteError must have at least one example before it can be computed.")
        return round(self._sum_of_errors_mae / self._num_examples_mae, 3)

    def rmse(self,output):
        """
        Calculates the mean squared error.
        - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
        Arguments:
            output(tuple): A tuple (prediction tensor, ground truth tensor)
        """
        y_pred, y = output
        squared_errors = torch.pow(torch.abs(y_pred - y.view_as(y_pred)), 2)
        self._sum_of_errors_rmse += torch.sum(squared_errors).item()
        self._num_examples_rmse += y.shape[0]

        if self._num_examples_rmse == 0:
            raise NotComputableError("RootMeanSquaredError must have at least one example before it can be computed.")
        return round(math.sqrt(self._sum_of_errors_rmse / self._num_examples_rmse), 3)


    def mare(self, output):
        """
        Calculate Mean Absolute Relative Error
        Arguments:
            output(tuple): A tuple (prediction tensor, ground truth tensor)
        """
        y_pred, y = output
        epsilon = 0.5
        absolute_error = torch.abs(y_pred - y.view_as(y_pred)) / (torch.abs(y.view_as(y_pred)) + epsilon)
        self._sum_of_errors_mare += torch.sum(absolute_error).item()
        self._num_examples_mare += y.size()[0]

        if self._num_examples_mare == 0:
            raise NotComputableError('MeanAbsoluteRelativeError must have at least'
                                     'one sample before it can be computed.')
        return round(self._sum_of_errors_mare / self._num_examples_mare, 3)

    def iou(self, output):
        """Calculate Intersection Over Union Error
        Arguments:
            output(tuple): A tuple (prediction tensor, ground truth tensor)
        """

        y_pred, y = output
        y_pred, y = y_pred.squeeze(1), y.squeeze(1)
        intersection = (y_pred * y).sum(2).sum(1)
        union = (y_pred + y).sum(2).sum(1) - intersection

        epsilon = 1e-6
        iou = (intersection + epsilon) / (union + epsilon)
        self._sum_of_errors_iou += iou.sum().item()
        self._num_examples_iou += y.shape[0]

        if self._num_examples_iou == 0:
            raise NotComputableError('IntersectionOverUnion must have at least'
                                     'one sample before it can be computed.')
        return round( self._sum_of_errors_iou/self._num_examples_iou, 3)

    def _pred_label_diff(self, label, prediction, rel=False):
        """Calculate the difference between label and prediction.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
            rel (bool, optional): If True, return the relative
                difference. (default: False)
        
        Returns:
            Difference between label and prediction
        """
        # For numerical stability
        valid_labels = label > 0.0001
        _label = label[valid_labels]
        _prediction = prediction[valid_labels]
        valid_element_count = _label.size(0)

        if valid_element_count > 0:
            diff = torch.abs(_label - _prediction)
            if rel:
                diff = torch.div(diff, _label)
            
            return diff, valid_element_count

    def _rmse(self, output):
        """Calculate Root Mean Square Error.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        
        Returns:
            Root Mean Square Error
        """
        prediction, label  = output
        diff = self._pred_label_diff(label, prediction)
        rmse = 0
        if not diff is None:
            rmse = math.sqrt(torch.sum(torch.pow(diff[0], 2)) / diff[1])
        
        self._sum_of_errors_rmse_new += label.size(0)
        self._num_examples_rmse_new += rmse * label.size(0)
        return round(
            self._sum_of_errors_rmse_new / self._num_examples_rmse_new, 3
        )