import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensor

import zipfile
import requests
from io import StringIO, BytesIO
from tqdm import tqdm

from deepnet.data.dataset.basedataset import BaseDataset

class TinyImageNet(BaseDataset):

    @property
    def mean(self):
        """Returns Channel-wise mean of the whole dataset"""
        return (0.4914, 0.4822, 0.4465) 


    @property
    def std(self):
        """Returns Channel-wise standard deviation of the whole dataset"""
        return (0.2023, 0.1994, 0.2010)


    @property
    def input_size(self):
        """Returns Dimension of the input image"""
        return (64, 64, 3)


    def download(self, train=True, apply_transformations=False):
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
            'path': self.path,
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
        self._train_transformations = self.transform(self.mean, self.std)
        train_data = self.download(train = True)

        self._test_transformations = self.transform(self.mean, self.std, train = False)
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
        self.path = path

        if self.path == None:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset')
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.download_dataset(self.path)

        #make a dictionary of filename and class name
        filename_to_class = {}
        self.path = os.path.join(self.path, 'tiny-imagenet-200')

        label_200 = os.path.join(self.path, 'wnids.txt')
        label_all = os.path.join(self.path, 'words.txt')

        with open(label_200, 'r') as t_label:
            for value in t_label.read().splitlines():
                filename_to_class[value] = 0

        with open(label_all, 'r') as a_label:
            for value in a_label.read().splitlines():
                filename = value.split('\t') 
                if filename[0] in filename_to_class:
                    image_class = filename[1].split(',')
                    filename_to_class[filename[0]] = image_class[0]

        #get data from test folder
        test_class_path = os.path.join(self.path, 'val/val_annotations.txt')

        testfilename_to_filename = {}
        with open(test_class_path, 'r') as f_label:
            for value in f_label.read().splitlines():
                value = value.split('\t')
                testfilename_to_filename[value[0]] =  filename_to_class[value[1]]

        #create dataset
        self.dataset = []
        train_path = os.path.join(self.path, 'train')
        test_path = os.path.join(self.path, 'val/images')

        for filename in os.listdir(train_path):
            _path = os.path.join(train_path, filename, 'images')
            for value in os.listdir(_path):
                self.dataset.append({
                    'image_path': os.path.join(_path,value),
                    'image_class':filename_to_class[filename]
                })
        
        for filename in os.listdir(test_path):
            self.dataset.append({
                'image_path': os.path.join(test_path, filename),
                'image_class':testfilename_to_filename[filename]
            })
        
        #class_to_index
        self.class_to_idx = {}
        index = 0

        for value in sorted((list(filename_to_class.values()))):
            self.class_to_idx[value] = index
            index += 1

        total_images = len(self.dataset)
        image_index = list(range(0,total_images))

        np.random.seed(seed)
        np.random.shuffle(image_index)
        
        last_train_index = int(total_images*train_test_split)
        
        if train:
            image_index = image_index[:last_train_index]
        else:
            image_index = image_index[last_train_index:]

        #stores path and class of the image
        self.data = []
        for index in image_index:
            data = self.dataset[index]
            img = Image.open(data['image_path'])
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            img = Image.fromarray(img)
            self.data.append({
                'image': img,
                'class': data['image_class'],
                'class_idx': self.class_to_idx[data['image_class']]
            })


    def __len__(self):
        '''Returns the length of the data'''
        return len(self.data)

    def __getitem__(self, idx):
        '''Return the data'''
        image = self.data[idx]['image']
        image = self.transforms(image)

        label = self.data[idx]['class_idx']

        return tuple((image,label))

    def download_dataset(self, path):
    '''Downloads the TinyImageNet dataset
    Arguments:
        path (str): Path to store the dataset
    '''

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    if os.path.isdir('tiny-imagenet-200.zip'):
        print('Images already downloaded!')
    r = requests.get(url, stream = True)
    print('Downloading '+url)
    with zipfile.ZipFile(BytesIO(r.content)) as zip:
        for member in tqdm(zip.infolist(), desc='Extracting'):
            try:
                zip.extract(member,path)
            except zipfile.error as e:
                pass
