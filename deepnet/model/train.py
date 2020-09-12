import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

from deepnet.model.losses.ssim import SSIM, MS_SSIM
from deepnet.model.losses.dice_loss import DiceLoss
from deepnet.utils.progress_bar import ProgressBar

class Train:
    def __init__(self):
        return

    def _fetch_data(self, batch, device):
        """Fetchs data and target from batch
        Arguments:
            batch: One batch from the dataloader
            device: Device on which model will be trained (GPU/CPU)
        """
        data, target = batch[0].to(device), batch[1].to(device)
        return data, target

    def _fetch_result(self, prediction):
        raise NotImplementedError

    def _fetch_sigmoid_data(self, prediction):
        raise NotImplementedError

    def _fetch_prediction(self, y_pred, criterion):
        if isinstance(y_pred, tuple):
            result = self._fetch_result(y_pred)
        elif isinstance(criterion, SSIM) or isinstance(criterion, MS_SSIM) or isinstance(criterion, DiceLoss):
            result = torch.sigmoid(y_pred)
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            result = y_pred
        return result

    def _fetch_metrics(
        self, y_pred, batch_idx, len_trainloader, phase, list_of_metrics, metric, target, loss, 
        tensorboard, epoch
        ):

        if isinstance(y_pred, tuple):
            output = self._fetch_sigmoid_data(y_pred)
        else:
            output = torch.sigmoid(y_pred)

        is_last_iter = False
        if batch_idx == len_trainloader-1:
            is_last_iter = True
        pkbar = self.metric_eval(
            phase, list_of_metrics, metric, output, target, loss, tensorboard, epoch, is_last_iter
        ) 

        return pkbar

    def _end_batch(self, is_metrics, pkbar, pbar, batch_idx, loss):

        pkbar_logs = []
        if is_metrics:
            if isinstance(pkbar, dict):
                for k,v in pkbar.items():
                    k = k.split('_')[-1]
                    for vdata in v:
                        rex = [('|'+k+'_'+vdata[0],vdata[1])]
                        pkbar_logs.extend(rex)
            else:
                pkbar_logs = pkbar

            pbar.update(batch_idx, values=pkbar_logs)
        else:
            pbar.update(batch_idx, values=[('loss', round(loss, 6))])
        return pkbar_logs

    def _end_epoch(self, pbar, is_metrics, metrics, pkbar, pkbar_logs, loss):

        if is_metrics:
            metrics.append(pkbar)
            pbar.add(1, values=pkbar_logs)
        else:
            pbar.add(1, values=[
                ('loss', round(loss, 6))
            ])
        

    def train(self, epoch, model, train_loader, device, optimizer, criterion, list_of_metrics, metric, metrics, losses, accuracies, scheduler, tensorboard):
        """Trains the images and prints the loss and accuracy of the model on train dataset
        Arguments:
            model: Network on which training data learns
            val_loader : Dataloader which allows training data to be iterable
            device : Machine on which model runs [gpu/cpu]
            criterion : Loss function
            losses (list): Store loss
            accuracies (list): Store accuracy
            scheduler_type (str): Scheduler name
            scheduler: Scheduler to be applied on the model
        """
        pbar = ProgressBar(target=len(train_loader), width=8)
        print(f'Epoch {epoch}:')
        model.train()
        correct = 0
        processed = 0
        len_trainloader = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, 0):
            # Get samples
            data, target = self._fetch_data(batch, device)

            # Set gradients to zero before starting backpropagation
            optimizer.zero_grad()

            # Predict output
            y_pred = model(data)
            result = self._fetch_prediction(y_pred, criterion)

            # Calculate loss
            loss = criterion(result, target)
            
            pkbar = {}
            if list_of_metrics:
                pkbar = self._fetch_metrics(
                    y_pred, batch_idx, len_trainloader, 'Train', list_of_metrics, metric, target, loss.item(), 
                    tensorboard, epoch
                )
                            
            # Perform backpropagation
            loss.backward()
            optimizer.step()

            if 'OneCycleLR' in scheduler:
                scheduler['OneCycleLR'].step()
                tensorboard.tbmetrics('Learning rate', scheduler['OneCycleLR'].get_last_lr()[0], epoch)

            # Update evaluation data
            if isinstance(y_pred,tuple):
                correct += 50
            else:
                pred = y_pred.argmax(dim=1, keepdim=False)
                correct += pred.eq(target).sum().item()
            processed += len(data)

            pkbar_logs = self._end_batch(list_of_metrics, pkbar, pbar, batch_idx, loss.item())

        tensorboard.tbimages(model,'result',epoch)
        losses.append(loss)
        accuracies.append(100. * correct / processed)

        self._end_epoch( pbar, list_of_metrics, metrics, pkbar, pkbar_logs, loss.item())
        

    def test(self, epoch, model, val_loader, device, criterion, list_of_metrics, metric, metrics, losses, accuracies, tensorboard):
        """Tests the images and prints the loss and accuracy of the model on test dataset
        Arguments:
            model: Network on which validation data predicts output
            val_loader : Dataloader which allows validation data to be iterable
            device : Machine on which model runs [gpu/cpu]
            losscriterion_function : Loss function
            losses (list): Store loss
            accuracies (list): Store accuracy
        Returns:
            Validation loss
        """
        model.eval()
        correct = 0
        val_loss = 0
        processed = 0
        len_testloader = len(val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data, target = self._fetch_data(batch, device)  # Get samples
                y_pred = model(data)  # Get trained model output

                result = self._fetch_prediction(y_pred, criterion)

                # Calculate loss
                loss = criterion(result, target).item()
                val_loss += loss # Sum up batch loss
                
                pkbar = {}
                if list_of_metrics:
                    pkbar = self._fetch_metrics(
                        y_pred, batch_idx, len_testloader, 'Test', list_of_metrics, metric, target, loss, 
                        tensorboard, epoch
                    )
                # Update evaluation data
                if isinstance(y_pred,tuple):
                    correct += 50
                else:
                    pred = y_pred.argmax(dim=1, keepdim=False)
                    correct += pred.eq(target).sum().item()
                processed += len(data)

        val_loss /= len(val_loader.dataset)
        val_loss = round(val_loss, 6)
        losses.append(val_loss)
        accuracies.append(100. * len(val_loader.dataset))
        
        if list_of_metrics:
            pkbar_logs = []
            if isinstance(pkbar, dict):
                for k,v in pkbar.items():
                    k = k.split('_')[-1]
                    for vdata in v:
                        rex = [('|'+k+'_'+vdata[0],vdata[1])]
                        pkbar_logs.extend(rex)
            else:
                pkbar_logs = pkbar

            metrics.append(pkbar)
            result = ''
            for res in pkbar_logs:
                result += f'{res[0]}: {res[1]} '
            print(f'\n Validation: {result}\n')
        else:
            print(f'\n Validation: loss: {round(val_loss, 6)}\n')

        return val_loss, correct/len(val_loader.dataset)

    
    def save_output(self, pred, epoch, phase):
        """Save image
        Arguments:
            pred(torch.tensor): Predicted tensor
            epoch(int): Current epoch
            phase(str): Training or Testing
        """

        rex = pred.detach().cpu()
        fig, axs = plt.subplots(4,4, figsize=(12,12))
        axs = axs.flatten()

        for idx, ax in enumerate(axs):
            img = rex[idx].squeeze(0)
            ax.axis('off')
            ax.imshow(img, cmap='gray')
        fig.savefig(f'img_{epoch}_{phase}', bbox_inches='tight')
        plt.close(fig)  

    def metric_eval(self, phase, metrics, metric, output, target, loss, tensorboard, epoch, is_last):
        if isinstance(output, dict) and isinstance(target, dict):
            pkbar = {}
            for k,_ in output.items():
                multi_output = k+'/'
                res = self.metric(phase, metrics, metric, output[k], target[k], loss, tensorboard, epoch, is_last, multi_output)
                pkbar[k] = res
            return pkbar
        else:
            return self.metric(phase, metrics, metric, output, target, loss, tensorboard, epoch, is_last,'')

    def metric(self, phase, metrics, metric, output, target, loss, tensorboard, epoch, is_last, multi_output):
        pkbar = [('loss', round(loss,6))]
        result = {f'Loss/{phase}' : round(loss,6) }

        if 'mae' in metrics:
            mae_value = metric.mae((output, target))
            pkbar.append(('mae', mae_value))

        if 'rmse' in metrics:
            rmse_value = metric._rmse((output, target))
            pkbar.append(('rmse', rmse_value))

        if 'mare' in metrics:
            mare_value = metric.mare((output, target))
            pkbar.append(('mare', mare_value))

        if 'iou' in metrics:
            iou_value = metric.iou((output, target))
            pkbar.append(('iou', iou_value))

        
        if is_last:
            if 'mae' in metrics:
                result[f'{multi_output}MAE/{phase}'] = mae_value
            if 'rmse' in metrics:
                result[f'{multi_output}RMSE/{phase}'] = rmse_value
            if 'mare' in metrics:
                result[f'{multi_output}MARE/{phase}'] = mare_value
            if 'iou' in metrics:
                result[f'{multi_output}IOU/{phase}'] = iou_value
            tensorboard.tbmetrics('metric', result, epoch)

        return pkbar