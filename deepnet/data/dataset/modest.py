import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensor

from zipfile import ZipFile
from tqdm import tqdm

from deepnet.data.dataset.basedataset import BaseDataset

class ModestMuseum(BaseDataset):

    @property
    def mean(self):
        """Returns Channel-wise mean of the whole dataset"""
        return dict({
            'bg':(0.40086, 0.46599, 0.53281),
            'bg_fg':(0.41221, 0.47368, 0.53431),
            'bg_fg_mask':(0.05207),
            'bg_fg_depth':(0.2981)
            })


    @property
    def std(self):
        """Returns Channel-wise standard deviation of the whole dataset"""
        return dict({
            'bg':(0.25451, 0.24249, 0.23615),
            'bg_fg':(0.25699, 0.24577, 0.24217),
            'bg_fg_mask':(0.21686),
            'bg_fg_depth':(0.11561)
        })


    @property
    def input_size(self):
        """Returns Dimension of the input image"""
        # channels, height, width = self._train_data[0][0].shape
        # return tuple((channels, height, width))
        return dict({
            'bg': (3,224,224),
            'bg_fg':(3,224,224),
            'bg_fg_mask':(1,224,224),
            'bg_fg_depth':(1,224,224)
        })
    

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
        args = {
            'path' : self.path,
            'train': train,
            'train_test_split': self.train_test_split,
            'seed': self.seed,
            'transforms': transform}
        return Download(**args)


    def dataset_creation(self):
        """Creates dataset
        Returns:
            Train dataset and Test dataset
        """
        self._train_transformations = {
            'bg':self.transform(self.mean['bg'], self.std['bg']),
            'bg_fg':self.transform(self.mean['bg_fg'], self.std['bg_fg']),
            'bg_fg_mask':self.transform(self.mean['bg_fg_mask'], self.std['bg_fg_mask'], modest_input=False),
            'bg_fg_depth':self.transform(self.mean['bg_fg_depth'], self.std['bg_fg_depth'], modest_input=False)
            }
        train_data = self.download(train = True)

        self._test_transformations = {
            'bg':self.transform(self.mean['bg'], self.std['bg'], train=False),
            'bg_fg':self.transform(self.mean['bg_fg'], self.std['bg_fg'], train=False),
            'bg_fg_mask':self.transform(self.mean['bg_fg_mask'], self.std['bg_fg_mask'], train=False, modest_input=False),
            'bg_fg_depth':self.transform(self.mean['bg_fg_depth'], self.std['bg_fg_depth'], train=False, modest_input=False)
            }
        test_data = self.download(train = False)
        return train_data, test_data


class Download(Dataset):
    def __init__(self, path, train=False, train_test_split=0.7, seed=1, transforms=None):
        '''Extract the data and target from the dataset folder
        Arguments:
            path (str): Path to store the dataset
            train (bool): True if train data is to be extracted, False is test data is to be extracted
                (default: False)
            train_test_split (float, optional) : Value to split train test data for training
                (default: 0.7)
            seed (integer, optional): Value for random initialization
                (default: 1)
            transforms: Transformations that are to be applied on the data
                (default: None)
        '''

        self.train = train
        self.transforms = transforms

        data = []
        file_map = open(os.path.join(path,'file_map.txt'))
        file_info = file_map.readlines()

        for f in file_info:
            mapping = f[:-1].split('\t')
            data.append({'bg' : os.path.join(path,'bg',mapping[0] + '.jpeg'),
                        'bg_fg' : os.path.join(path,'bg_fg',mapping[1] + '.jpeg'),
                        'bg_fg_mask' : os.path.join(path,'bg_fg_mask',mapping[2] + '.jpeg'),
                        'bg_fg_depth' : os.path.join(path,'bg_fg_depth_map',mapping[3] + '.jpeg'),})

        total_images = len(data)
        image_index = list(range(0,total_images))

        np.random.seed(seed)
        np.random.shuffle(image_index)

        last_train_index = int(total_images*train_test_split)

        if train:
            image_index = image_index[:last_train_index]
        else:
            image_index = image_index[last_train_index:]

        #stores path and class of the image
        self.dataset = []
        for index in image_index:
            self.dataset.append(data[index])

    def __len__(self):
        '''Returns the length of the data'''
        return len(self.dataset)

    def __getitem__(self, idx):
        '''Return the data'''

        data = self.dataset[idx]
        bg = self.transforms['bg'](Image.open(data['bg']))
        bg_fg = self.transforms['bg_fg'](Image.open(data['bg_fg']))
        bg_fg_mask = self.transforms['bg_fg_mask'](Image.open(data['bg_fg_mask']))
        bg_fg_depth = self.transforms['bg_fg_depth'](Image.open(data['bg_fg_depth']))

        data = {
            'bg' : bg,
            'bg_fg' : bg_fg,
            'bg_fg_mask' : bg_fg_mask,
            'bg_fg_depth' : bg_fg_depth
        }
        return data