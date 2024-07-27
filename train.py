import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer

parser = argparse.ArgumentParser()
# Basic parameters
parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet'])  # The network architecture
parser.add_argument('--dataset', type=str, default='miniImageNet',
                choices=['miniImageNet', 'CIFAR-FS', 'FC100'])  # Dataset
parser.add_argument('--phase', type=str, default='meta_train',
                choices=['pre_train', 'meta_train', 'meta_eval', 'OOD_test', 'threshold_test',
                         'active'])  # Phase
parser.add_argument('--seed', type=int, default=0)  # Manual seed for PyTorch, "0" means using random seed
parser.add_argument('--gpu', default='0')  # GPU id

# Parameters for meta-train phase
parser.add_argument('--max_epoch', type=int, default=100)  # Epoch number for meta-train phase
parser.add_argument('--train_num_batch', type=int, default=200)
parser.add_argument('--val_num_batch', type=int, default=600)
parser.add_argument('--shot', type=int, default=5)  # Shot number, how many samples for one class in a task
parser.add_argument('--way', type=int, default=5)  # Way number, how many classes in a task
parser.add_argument('--train_query', type=int,
                default=15)  # The number of training samples for each class in a task
parser.add_argument('--val_query', type=int, default=15)  # The number of test samples for each class in a task
parser.add_argument('--active_query', type=int, default=35)  # The number of test samples for each class in a task
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

# Set the GPU id
set_gpu(args.gpu)

trainer = MetaTrainer(args)
trainer.meta_train()