""" Additional utility functions. """
import os
import time
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


class Averager():
    """The class to calculate the average."""
    
    def __init__(self):
        self.n = 0
        self.v = 0
    
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    
    def data(self):
        return self.v


def count_acc(input, label, logit):
    if logit:
        pred = F.softmax(input, dim=1).argmax(dim=1)
    else:
        pred = input.argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item() * 100
    return (pred == label).type(torch.FloatTensor).mean().item() * 100


class Timer():
    """The class for timer."""
    
    def __init__(self):
        self.o = time.time()
    
    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins, logit):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.logit = logit

    def forward(self, inputs, labels):
        labels = labels
        if self.logit:
            probs = F.softmax(inputs, dim=1)
        else:
            probs = inputs
        # torch.max(p, 1): 返回每行的最大值和最大值的列索引
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            # tensor.gt: great than
            # tensor.le: less or equal
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def brier_score(inputs, labels, logit):
    if logit:
        probs = F.softmax(inputs, dim=1)
    else:
        probs = inputs
    bs = torch.mean(torch.sum((labels - probs) ** 2, dim=1))
    return bs


def get_entropy(inputs, logits):
    if logits:
        prob = F.softmax(inputs, dim=1)
    else:
        prob = inputs
    ent = -torch.sum(prob * torch.log(prob), dim=1)
    return ent

def plot_ECDF(ent):
    e_sorted, _ = torch.sort(ent)
    ECDF = torch.arange(1., len(ent) + 1) / len(ent)
    plt.plot(e_sorted, ECDF)
    plt.xlabel('Entropy')
    plt.ylabel('ECDF')
    plt.show()


def calc_variance(inputs, logit):
    if logit:
        probs = F.softmax(inputs, dim=1)
    else:
        probs = inputs
    variance = torch.var(probs, dim=1)
    return variance

def get_confidence(inputs, logit):
    if logit:
        prob = F.softmax(inputs, dim=1)
    else:
        prob = inputs
    confidence, prediction = torch.max(prob, 1)
    return prediction, confidence

def get_task_data(task, args):
    data = task[0].cuda()
    p = args.shot * args.way
    data_support, data_query = data[:p], data[p:]

    label_support = torch.arange(args.way).repeat(args.shot)
    label_support = label_support.type(torch.cuda.LongTensor)

    # Generate the labels for test set of the episodes during meta-train updates
    label_query = torch.arange(args.way).repeat(args.query)
    label_query = label_query.type(torch.cuda.LongTensor)

    return data_support, label_support, data_query, label_query

