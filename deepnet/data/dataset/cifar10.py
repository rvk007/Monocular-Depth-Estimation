import numpy as np
from torchvision import datasets

from deepnet.data.dataset.basedataset import BaseDataset

class CIFAR10(BaseDataset):

    @property
    def test_data(self):
        """Returns Test Dataset"""
        return self._test_data


    @property
    def mean(self):
        """Returns Channel-wise mean of the whole dataset"""
        return (0.49139, 0.48215, 0.44653)


    @property
    def std(self):
        """Returns Channel-wise standard deviation of the whole dataset"""
        return (0.24703, 0.24348, 0.26158)


    @property
    def input_size(self):
        """Returns Dimension of the input image"""
        return (32, 32, 3)


    def download(self, train=True):
        """Download the dataset
        Arguments:
            train: True if train data is to downloaded, False for test data
                (default: True)
            apply_transformations: True if transformations is to applied, else False
                (default: False)
        Returns:
            Dataset after downloading
        """
        transform = self._train_transformations if train else self._test_transformations
        # Download data
        return datasets.CIFAR10('./data', train=train, download=True, transform = transform)
    

    def dataset_creation(self):
        """Creates dataset
        Returns:
            Train dataset and Test dataset
        """
        self._train_transformations = self.transform(self.mean, self.std)
        train_data = self.download(train = True)

        self._test_transformations = self.transform((self.mean, self.std, train = False)
        test_data = self.download(train = False)
        return train_data, test_data

