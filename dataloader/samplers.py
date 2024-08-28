""" Sampler for dataloader. """
import torch
import numpy as np


class CategoriesSampler:
    """The class to generate episodic data"""
    
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        
        # label = [ 0  0  0 ... 15 15 15]
        # len(label) = 600 * len(classes)
        label = np.array(label)
        # m_ind: a list with shape 16*600
        self.m_ind = []
        for i in range(min(label) + 1, max(label) + 1, 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 先随机打乱所有类，再从所有类里面挑选出前N个类
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            # 对N个类中的每一个类
            for c in classes:
                # l = 第c个类样本所在的index
                l = self.m_ind[c]
                # 先打乱第c个类的所有样本，然后再选出所需的样本数
                pos = torch.randperm(len(l))[:self.n_per]
                # 每个batch相当于从所有类中选出N个类，然后每个类选出K+Q个样本
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # 每个batch返回的是一个大小为N*(K+Q)的列表，列表里的每一个元素对应Dataset里面的元素的index
            # 通过这个index，可以得到对应的图片和标签
            yield batch
