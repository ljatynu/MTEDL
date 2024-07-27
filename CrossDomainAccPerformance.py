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


parser = argparse.ArgumentParser()
# Basic parameters
parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet'])  # The network architecture
parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'Places'])  # Dataset
parser.add_argument('--phase', type=str, default='meta_eval', choices=['pre_train', 'meta_train', 'meta_eval', 'OOD_test', 'threshold_test', 'active'])  # Phase
parser.add_argument('--seed', type=int, default=0)  # Manual seed for PyTorch, "0" means using random seed
parser.add_argument('--gpu', default='0')  # GPU id

# Parameters for meta-train phase
parser.add_argument('--max_epoch', type=int, default=100)  # Epoch number for meta-train phase
parser.add_argument('--train_num_batch', type=int, default=200)
parser.add_argument('--val_num_batch', type=int, default=600)
parser.add_argument('--way', type=int, default=5)  # Way number, how many classes in a task
parser.add_argument('--shot', type=int, default=1)  # Shot number, how many samples for one class in a task
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

parser.add_argument('--pretrain_evidence_weight', type=int, default=100)
parser.add_argument('--meta_evidence_weight', type=int, default=1000)
parser.add_argument('--annealing_step', type=int, default=100)

# Set and print the parameters
args = parser.parse_args()
pprint(vars(args))
set_gpu(args.gpu)

print('==> Preparing In-distribution data...')
if args.dataset == 'CUB':
    from data.CUB import CUB, FewShotDataloader
    testdataset = CUB(phase='test')
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
elif args.dataset == 'Places':
    from data.Places import Places, FewShotDataloader
    testdataset = Places(phase='test')
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

print('==> Preparing Model...')
model = MtlLearner(args)

# Load the meta-trained model
args.save_path = 'checkpoints/meta/miniImageNet-{}-shot/max_acc.pth'.format(args.shot)
print('==> Loading meta-training model from: ', args.save_path)
model.load_state_dict(torch.load(args.save_path)['params'])
model.to('cuda')
model.eval()

accs = []
eces = []

ECE = ECELoss(n_bins=15, logit=False)

for task_id, task in enumerate(test_loader(1)):
    data_support, label_support, data_query, label_query, _, _ = [x.squeeze().cuda() for x in task]

    evidence = model.cross_domain_forward(data_support, label_support, data_query)

    alpha = evidence + 1

    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    log_alpha = torch.log(alpha)

    acc = count_acc(prob.detach(), label_query, logit=False)

    accs.append(acc)

    avg_acc, std_acc, ci95_acc = calculate_avg_std_ci95(accs)

    print('Cross-Domain classification performance: Task [{}/{}]: Accuracy: {:.2f} ± {:.1f} % ({:.2f} %)'.
        format(task_id, len(test_loader), avg_acc, ci95_acc, acc))

    pass
