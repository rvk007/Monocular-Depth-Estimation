import torch
import os
import numpy as np

class Checkpoint:
    def __init__(self, path, monitor='Accuracy', mode='max', save_best_only=False, verbose=0, last_reload=False):

        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.last_reload = last_reload

        self.last_checkpoint = None
        if self.last_reload:
            self.last_checkpoint = self.reload_checkpoint()

        self.last_metric = self.checkpoint_metric()

    def reload_checkpoint(self):
        if os.path.exists(self.path):
            last_checkpoint = torch.load(self.path,
                                    map_location=lambda storage,
                                    loc: storage)
            return last_checkpoint
        else:
            print(f'{self.path}: No such file exists.')

    def checkpoint_metric(self):
        if not os.path.exists(self.path):
            root = os.path.abspath(__file__)
            path = os.path.join(root, self.path)
            x = path.split('/')
            if x[-1].endswith('.pth'):
                os.mkdir(x[-2])
                os.mknod(os.path.join(x[-2], x[-1]))

        if self.monitor == 'Accuracy':
            metric = -np.Inf
            mode = 'max'
        elif self.monitor == 'Loss':
            metric = np.Inf
            mode = 'min'
        return metric, mode

    def save(self, model_state):
        torch.save(model_state, self.path)
