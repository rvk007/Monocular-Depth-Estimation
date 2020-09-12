import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR

from deepnet.utils.checkpoint import Checkpoint
from deepnet.model.metrics import Metric


class Model:
    def __init__(self, start_epoch, model, tensorboard, dataset, train_loader, test_loader, device, optimizer, criterion, 
        epochs, metrics, callbacks=None
    ):  
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
        self.start_epoch = start_epoch
        self.model=model
        self.tensorboard = tensorboard
        self.dataset = dataset
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.device=device
        self.optimizer=optimizer
        self.criterion=criterion
        self.epochs=epochs
        self.metrics = metrics
        self.callbacks=callbacks

    def fit(self):
        """Returns the training and validation results"""

        self.train_losses = []
        self.train_accuracies = []
        self.train_metrics = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_metrics = []
        self.last_saved_epoch = 0
        
        self.scheduler = {}
        self.checkpoint = None
        self.initialize_callbacks()

        if self.checkpoint:
            self.last_metric, self.mode = self.checkpoint.checkpoint_metric()
            self.verbose = self.checkpoint.verbose

        # model_input = {}
        # for batch in self.train_loader:
        #     model_input['bg'] = batch['bg'].to(self.device)
        #     model_input['bg_fg'] = batch['bg_fg'].to(self.device)
        #     break

        # self.tensorboard.tbmodel(self.model, model_input)
        
        for epoch in range(1, self.epochs + 1):
            if self.start_epoch:
                epoch += self.start_epoch
                
            if self.metrics:
                metric = Metric()

            if self.checkpoint.last_reload:
                self.reload_last_checkpoint()

            self.dataset.train(
                epoch, self.model, self.train_loader, self.device, self.optimizer, self.criterion,
                self.metrics, metric, self.train_metrics, self.train_losses,self.train_accuracies,
                self.scheduler, self.tensorboard
            )

            if 'StepLR' in self.scheduler:
                self.scheduler['StepLR'].step()
                self.tensorboard.tbmetrics('Learning rate', self.scheduler['StepLR'].get_last_lr()[0], epoch)

            if self.metrics:
                metric.reset()
                
            val_loss, val_acc = self.dataset.test(
                epoch, self.model, self.test_loader, self.device, self.criterion, self.metrics, 
                metric, self.test_metrics, self.test_losses, self.test_accuracies, self.tensorboard
            )

            if 'ReduceLROnPlateau' in self.scheduler:
                self.scheduler['ReduceLROnPlateau'].step(val_loss)
                self.tensorboard.tbmetrics('Learning rate', self.scheduler['ReduceLROnPlateau']._last_lr[0], epoch)

            if self.ischeckpoint(epoch, val_acc, val_loss):
                state = {
                    'epoch' : self.last_saved_epoch + epoch + 1,
                    'accuracy' : val_acc,
                    'loss' : val_loss,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                }

                self.checkpoint.save(state)
            
            if self.metrics:
                metric.reset()

        result = {
            'train_loss' : self.train_losses,
            'train_accuracy': self.train_accuracies,
            'train_metrics': self.train_metrics,
            'test_loss': self.test_losses,
            'test_accuracy': self.test_accuracies,
            'test_metrics': self.test_metrics}
        return result

        
    def initialize_callbacks(self):
        """Initailize all the callbacks given to the model"""

        for callback in self.callbacks:
            if isinstance(callback, torch.optim.lr_scheduler.StepLR):
                self.scheduler['StepLR'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler['ReduceLROnPlateau'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler['OneCycleLR'] = callback
            elif isinstance(callback, Checkpoint):
                self.checkpoint = callback
    
    def ischeckpoint(self, epoch, accuracy, loss):
        """Initialize checkpoint
        Arguments:
            epoch: Current epoch
            accuracy: Validation accuracy for this epoch
            loss: Validation loss for this epoch
        """

        monitor = self.checkpoint.monitor

        if self.checkpoint.save_best_only:
            if self.mode == 'max':
                if accuracy > self.last_metric:
                    self.last_metric = accuracy
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, accuracy))
                    return True
            elif self.mode == 'min':
                if loss < self.last_metric:
                    self.last_metric = loss
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, loss))
                    return True
            if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, monitor, self.last_metric))
            return False
        else:
            if self.verbose > 0:
                if monitor == 'Accuracy' and accuracy > self.last_metric:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, accuracy))
                    self.last_metric = accuracy

                elif monitor == 'Loss' and loss < self.last_metric:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, loss))
                    self.last_metric = loss
                    
                else:
                    print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, monitor, self.last_metric))

            return True

    def reload_last_checkpoint(self):
        """Reload last checkpoint"""
        data = self.checkpoint.last_checkpoint

        self.last_saved_epoch = data['epoch']
        self.last_accuracy = data['accuracy']
        self.last_loss = data['loss']
        self.model.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
