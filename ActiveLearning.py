import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.CIFAR_FS import CIFAR_FS
from dataloader.FC100 import FC100
from dataloader.miniImageNet import MiniImageNet
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import pprint, count_acc, get_task_data
from utils.gpu_tools import set_gpu

parser = argparse.ArgumentParser()
# Basic parameters
parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet'])  # The network architecture
parser.add_argument('--dataset', type=str, default='CIFAR-FS', choices=['miniImageNet', 'CIFAR-FS', 'FC100'])  # Dataset
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
parser.add_argument('--active_query', type=int, default=35)  # The number of test samples for each class in a task
parser.add_argument("--task_num", default=600, type=int, help="Number of test tasks")

parser.add_argument('--meta_lr1', type=float, default=0.0001)  # Learning rate for SS weights
parser.add_argument('--meta_lr2', type=float, default=0.001)  # Learning rate for FC weights
parser.add_argument('--meta_base_lr', type=float, default=0.01)  # Learning rate for the inner loop
parser.add_argument('--pre_base_lr', type=float, default=0.01)  # Learning rate for the inner loop
parser.add_argument('--update_step', type=int, default=200)  # The number of updates for the inner loop
parser.add_argument('--step_size', type=int, default=10)  # The number of epochs to reduce the meta learning rates
parser.add_argument('--gamma', type=float, default=0.5)  # Gamma for the meta-train learning rate decay
parser.add_argument('--loss_type', type=str, default='log', choices=['mse', 'log', 'digamma'])

# Set and print the parameters
args = parser.parse_args()
pprint(vars(args))
set_gpu(args.gpu)

args.query = args.query + args.active_query


print('==> Preparing In-distribution data...')
if args.dataset == 'miniImageNet':
    Dataset = MiniImageNet
elif args.dataset == 'CIFAR-FS':
    Dataset = CIFAR_FS
elif args.dataset == 'FC100':
    Dataset = FC100
# Load meta-train set
dataset = Dataset('test')
sampler = CategoriesSampler(dataset.label, args.task_num, args.way,
                            args.shot + args.query)
dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, pin_memory=True)

print('==> Preparing Model...')
model = MtlLearner(args)

# Load the meta-trained model
args.save_path = 'checkpoints/meta/{}-{}-shot/max_acc.pth'.format(args.dataset, args.shot)
print('==> Loading meta-training model from: ', args.save_path)
model.load_state_dict(torch.load(args.save_path)['params'])
model.to('cuda')
model.eval()

active_accs = []
random_accs = []

# Generate labels
label_shot = torch.arange(args.way).repeat(args.shot)
if torch.cuda.is_available():
    label_shot = label_shot.type(torch.cuda.LongTensor)
else:
    label_shot = label_shot.type(torch.LongTensor)

# Start meta-test
total_ID_u = []
total_OOD_u = []
_, task = next(enumerate(dataloader))

data_support, label_support, data_query, label_query = get_task_data(task, args)

# top k per class
k = 5

logits_q, evidence_q, meta_fw, pre_fw = model.active_forward(data_support, label_support, data_query, epoch=0)

# update the parameter
model.meta_base_learner.fc1_w.data = meta_fw[0]
model.meta_base_learner.fc1_b.data = meta_fw[1]
model.pre_base_learner.fc1_w.data = pre_fw[0]
model.pre_base_learner.fc1_b.data = pre_fw[1]

alpha_q = evidence_q + 1

u_q = (5 / torch.sum(alpha_q, dim=1, keepdim=True)).view(-1)
u = u_q.reshape(-1, args.way)
u_t = torch.transpose(u, dim0=0, dim1=1)
u_random = torch.rand(u_t.shape)

Q_set = data_query.reshape(-1, args.way, 3, 84, 84)
Q = torch.transpose(Q_set, dim0=0, dim1=1)

label_set = label_query.reshape(-1, args.way)
label_Q = torch.transpose(label_set, dim0=0, dim1=1)
value_max, index_max = torch.topk(u_t, k, dim=1)
value_max_random, index_max_random = torch.topk(u_random, k, dim=1)
index_test_active = u_t < value_max[..., -1, None]
index_test_random = u_random < value_max_random[..., -1, None]
index_train_active = ~index_test_active
index_train_random = ~index_test_random
train_data_active = Q[index_train_active]
train_data_random = Q[index_train_random]
train_label_active = label_Q[index_train_active]
train_label_random = label_Q[index_train_random]
test_data_active = Q[index_test_active]
test_data_random = Q[index_test_random]
test_label_active = label_Q[index_test_active]
test_label_random = label_Q[index_test_random]

# re-train
logits_active_q, evidence_activate_q, _, _ = model.active_forward(
    train_data_active, train_label_active, test_data_active,
    epoch=args.max_epoch)
acc_activate = count_acc(logits_active_q, test_label_active, logit=None)

# model.zero_grad()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
# del logits_active_q, train_data_active, train_label_active, test_data_active, test_label_active

logits_random_q, evidence_random_q, _, _ = model.active_forward(
    train_data_random, train_label_random, test_data_random,
    epoch=args.max_epoch)

acc_random = count_acc(logits_random_q, test_label_random, logit=None)
del logits_random_q, train_data_random, train_label_random, test_data_random, test_label_random

print('Active Learning Performance')
print('Active acc:', acc_activate)
print('Random acc:', acc_random)