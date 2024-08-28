import os
import pickle

from torch.utils.data import Dataset


_PLACES_DATASET_DIR = 'datasets/Places'


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


class Places(Dataset):
    """The class to load the dataset"""

    def __init__(self, phase='train'):
        data_path = os.path.join(_PLACES_DATASET_DIR, f'Places_{phase}.pickle')

        data = load_data(data_path)
        self.data = data['data']
        self.label = data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        # image.shape = [3, 84, 84]
        return data, label