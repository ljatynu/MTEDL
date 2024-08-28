""" Dataloader for all datasets. """
import os.path as osp
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


_CIFAR_FS_DATASET_DIR = 'datasets/CIFAR-FS'

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


class CIFAR_FS(Dataset):
    """The class to load the dataset"""
    
    def __init__(self, phase='train'):
        data_path = os.path.join(_CIFAR_FS_DATASET_DIR, f'CIFAR_FS_{phase}.pickle')

        data = load_data(data_path)
        self.data = data['data']
        self.label = data['labels']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        # image.shape = [3, 84, 84]
        return data, label
