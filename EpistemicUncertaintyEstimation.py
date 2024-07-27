import argparse

import numpy as np
import torch
import os.path as osp

from metrics import compute_differential_entropy, compute_mutual_information, compute_precision, ROC_OOD
from models.mtl import MtlLearner
from utils.misc import pprint, ECELoss, count_acc
from utils.gpu_tools import set_gpu

def calculate_avg_std_ci95(data):
    avg = np.mean(np.array(data))
    std = np.std(np.array(data))
    ci95 = 1.96 * std / np.sqrt(len(data))

    return avg, std, ci95


def acc_with_threshold(preds, labels, uncertainty, threshold):
    under_threshold_index = uncertainty <= threshold
    preds_filter = preds[under_threshold_index]
    labels_filter = labels[under_threshold_index]
    filter_nums = len(labels_filter)
    if filter_nums == 0:
        return 0, 0
    else:
        match = torch.eq(preds_filter, labels_filter).float()
        acc_nums = torch.sum(match)
        return int(acc_nums), filter_nums

parser = argparse.ArgumentParser()
# Basic parameters
parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet'])  # The network architecture
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'CIFAR-FS', 'FC100'])  # Dataset
parser.add_argument('--phase', type=str, default='meta_eval', choices=['pre_train', 'meta_train', 'meta_eval', 'OOD_test', 'threshold_test', 'active'])  # Phase
parser.add_argument('--seed', type=int, default=0)  # Manual seed for PyTorch, "0" means using random seed
parser.add_argument('--gpu', default='0')  # GPU id

# Parameters for meta-train phase
parser.add_argument('--max_epoch', type=int, default=100)  # Epoch number for meta-train phase
parser.add_argument('--train_num_batch', type=int, default=200)
parser.add_argument('--val_num_batch', type=int, default=600)
parser.add_argument('--way', type=int, default=5)  # Way number, how many classes in a task
parser.add_argument('--shot', type=int, default=5)  # Shot number, how many samples for one class in a task
parser.add_argument('--query', type=int, default=15)  # The number of training samples for each class in a task
parser.add_argument("--task_num", default=600, type=int, help="Number of test tasks")

parser.add_argument('--meta_lr1', type=float, default=0.0001)  # Learning rate for SS weights
parser.add_argument('--meta_lr2', type=float, default=0.001)  # Learning rate for FC weights
parser.add_argument('--meta_base_lr', type=float, default=0.01)  # Learning rate for the inner loop
parser.add_argument('--pre_base_lr', type=float, default=0.01)  # Learning rate for the inner loop
parser.add_argument('--update_step', type=int, default=200)  # The number of updates for the inner loop
parser.add_argument('--step_size', type=int, default=10)  # The number of epochs to reduce the meta learning rates
parser.add_argument('--gamma', type=float, default=0.5)  # Gamma for the meta-train learning rate decay
parser.add_argument('--loss_type', type=str, default='log', choices=['mse', 'log', 'digamma'])

parser.add_argument('--threshold', type=float, default=0.1)


# Set and print the parameters
args = parser.parse_args()
pprint(vars(args))
set_gpu(args.gpu)

print('==> Preparing In-distribution data...')
if args.dataset=='miniImageNet':
    from data.mini_imagenet import MiniImageNet, FewShotDataloader
    testdataset = MiniImageNet(phase='test')
    test_loader = FewShotDataloader(
        dataset=testdataset,
        nKnovel=args.way,
        nKbase=0,
        nExemplars=args.shot,  # num training examples per novel category
        nTestNovel=args.way * args.query,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * args.task_num  # 600 test tasks
    )
elif args.dataset=='CIFAR-FS':
    from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
    testdataset = CIFAR_FS(phase='test')
    test_loader = FewShotDataloader(
        dataset=testdataset,
        nKnovel=args.way,
        nKbase=0,
        nExemplars=args.shot,  # num training examples per novel category
        nTestNovel=args.way * args.query,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * args.task_num  # 600 test tasks
    )

    pass
elif args.dataset=='FC100':
    from data.FC100 import FC100, FewShotDataloader

    testdataset = FC100(phase='test')
    test_loader = FewShotDataloader(
        dataset=testdataset,
        nKnovel=args.way,
        nKbase=0,
        nExemplars=args.shot,  # num training examples per novel category
        nTestNovel=args.way * args.query,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * args.task_num  # 600 test tasks
    )

print('==> Preparing Model...')
model = MtlLearner(args)

# Load the meta-trained model
args.save_path = 'checkpoints/meta/{}-5-shot/max_acc.pth'.format(args.dataset)
print('==> Loading meta-training model from: ', args.save_path)
model.load_state_dict(torch.load(args.save_path)['params'])
model.to('cuda')
model.eval()

total_acc_nums = 0
total_filter_nums = 0

for task_id, task in enumerate(test_loader(1)):
    data_support, label_support, data_query, label_query, _, _ = [x.squeeze().cuda() for x in task]

    evidence = model.threshold_forward(data_support, label_support, data_query, epoch=100)

    alpha = evidence + 1

    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)


    u = (args.way / torch.sum(alpha, dim=1, keepdim=True)).view(-1)

    _, preds = torch.max(prob, 1)

    acc_nums, filter_nums = acc_with_threshold(preds, label_query, u, args.threshold)

    total_acc_nums += acc_nums
    total_filter_nums += filter_nums

    if total_filter_nums == 0:
        filter_acc = 0
    else:
        filter_acc = total_acc_nums / total_filter_nums

    print(
        "Epistemic uncertainty Estimation: Task [{}/{}]: threshold={:.3f},acc={:.4f},total_acc_nums={},total_filter_nums={}".format(
            task_id, len(test_loader),
            args.threshold,
            filter_acc,
            total_acc_nums,
            total_filter_nums
        ))

    pass
