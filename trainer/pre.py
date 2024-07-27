""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, ensure_path



class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):
        log_base_dir = 'logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        # pre_base_dir = './logs/pre'
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        # save_path1 = MiniImageNet_ResNet
        save_path1 = '_'.join([args.dataset, args.model_type])
        # save_path2 = batchsize128_lr0.1_gamma_0.2_step30_maxepoch100
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + \
                     '_step' + str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) + '_shot' + str(
            args.shot)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        if args.dataset == 'miniImageNet':
            from data.mini_imagenet import MiniImageNet, FewShotDataloader
            self.trainset = MiniImageNet(phase='train')
            self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size,
                                       shuffle=True, pin_memory=True)
            self.valset = MiniImageNet(phase='val')
            self.val_loader = FewShotDataloader(
                                                dataset=self.valset,
                                                nKnovel=args.way,
                                                nKbase=0,
                                                nExemplars=args.shot,  # num training examples per novel category
                                                nTestNovel=args.way * args.query,
                                                # num test examples for all the novel categories
                                                nTestBase=0,  # num test examples for all the base categories
                                                batch_size=1,
                                                num_workers=0,
                                                epoch_size=600  # 600 test tasks
                                            )
        elif args.dataset == 'FC100':
            from data.FC100 import FC100, FewShotDataloader
            self.trainset = FC100(phase='train')
            self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size,
                                           shuffle=True, pin_memory=True)
            self.valset = FC100(phase='val')
            self.val_loader = FewShotDataloader(
                                                dataset=self.valset,
                                                nKnovel=args.way,
                                                nKbase=0,
                                                nExemplars=args.shot,  # num training examples per novel category
                                                nTestNovel=args.way * args.val_query,
                                                # num test examples for all the novel categories
                                                nTestBase=0,  # num test examples for all the base categories
                                                batch_size=1,
                                                num_workers=0,
                                                epoch_size=600  # 600 test tasks
                                            )

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_cats
        print('num_class_pretrain: ', num_class_pretrain)

        # Build pretrain model
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain)

        # Set optimizer 
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr},
                                          {'params': self.model.pre_fc.parameters(), 'lr': self.args.pre_lr}],
                                         momentum=self.args.pre_custom_momentum, nesterov=True,
                                         weight_decay=self.args.pre_custom_weight_decay)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size,
                                                            gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {'args': vars(self.args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
                 'max_acc': 0.0, 'max_acc_epoch': 0}

        # Set the timer
        timer = Timer()

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader
            # 总迭代次数 = 总样本数 / batch_size
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                # Output logits for model
                logits = self.model(data)
                # Calculate train loss
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label, logit=True)
                # Write the tensorboardX records
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.data()
            train_acc_averager = train_acc_averager.data()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Print previous information  
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for task_id, task in enumerate(self.val_loader(1)):
                data_support, label_support, data_query, label_query, _, _ = [x.squeeze().cuda() for x in task]

                logits = self.model((data_support, label_support, data_query))
                loss = F.cross_entropy(logits, label_query)
                acc = count_acc(logits, label_query, logit=True)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.data()
            val_acc_averager = val_acc_averager.data()
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 20 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
            self.lr_scheduler.step()
            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
