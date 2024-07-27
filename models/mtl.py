""" Model for meta-transfer learning. """
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

from models.EDL_loss import edl_mse_loss, edl_digamma_loss, edl_log_loss, softplus_evidence
from models.resnet12 import ResNet12Backbone
from models.resnet12_mtl import ResNet12Backbone_MTL


class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        self.vars.append(self.fc1_w)
        torch.nn.init.kaiming_uniform_(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        logits = F.linear(input_x, fc1_w, fc1_b)
        return logits

    def parameters(self):
        return self.vars


class MtlLearner(nn.Module):
    """The class for outer loop."""

    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_step = args.update_step
        z_dim = 640

        if self.mode == 'meta' or self.mode == 'meta_train':
            self.encoder = ResNet12Backbone_MTL()
            self.meta_update_lr = args.meta_base_lr
            self.pre_update_lr = args.pre_base_lr
            self.pre_encoder = ResNet12Backbone()
            self.pre_encoder.requires_grad_(requires_grad=False)
            self.pre_base_learner = BaseLearner(args, z_dim)
            self.meta_base_learner = BaseLearner(args, z_dim)
            if self.args.loss_type == 'mse':
                self.loss = edl_mse_loss
            elif self.args.loss_type == 'log':
                self.loss = edl_log_loss
            elif self.args.loss_type == 'digamma':
                self.loss = edl_digamma_loss
        else:
            self.encoder = ResNet12Backbone()
            self.pre_fc = nn.Sequential(nn.Linear(z_dim, 1000), nn.ReLU(), nn.Linear(1000, num_cls))
            self.base_learner = BaseLearner(args, z_dim)

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode == 'pre':
            return self.pretrain_forward(inp)
        elif self.mode == 'preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        elif self.mode == 'meta_train':
            data_shot, label_shot, data_query, epoch_num = inp
            return self.meta_train_forward(data_shot, label_shot, data_query, epoch_num)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        return self.pre_fc(self.encoder(inp))

    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_shot = self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        # fast_weights = self.base_learner.parameters() - 0.01 * grad
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            # fast_weights = fast_weights - 0.01 * grad
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q


    def meta_train_forward(self, data_shot, label_shot, data_query, epoch):
        # ANIL MODE
        # embeddings
        meta_embedding_shot = self.encoder(data_shot)
        meta_embedding_query = self.encoder(data_query)
        pre_embedding_shot = self.pre_encoder(data_shot)
        pre_embedding_query = self.pre_encoder(data_query)

        # base-learner outputs
        meta_logit = self.meta_base_learner(meta_embedding_shot)
        pre_logit = self.pre_base_learner(pre_embedding_shot)

        # losses
        meta_loss = F.cross_entropy(meta_logit, label_shot)
        pre_loss = F.cross_entropy(pre_logit, label_shot)

        # ANIL mode
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())

        # fast_weights, ANIL mode
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))

        # inner loop, ANIL mode
        for i in range(1, self.args.update_step):
            if i <= int(0.5 * self.args.update_step):
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_loss = F.cross_entropy(meta_logit, label_shot)
                pre_loss = F.cross_entropy(pre_logit, label_shot)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
            else:
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_evidence = softplus_evidence(meta_logit)
                pre_evidence = softplus_evidence(pre_logit)
                pre_alpha = 10 * pre_evidence + 1
                meta_alpha = 10 * meta_evidence + 1
                # total_evidence = 10*pre_evidence + 10*meta_evidence
                # total_alpha = total_evidence + 1
                pre_loss = self.loss(alpha=pre_alpha, target=label_shot, epoch_num=epoch,
                                     num_classes=5, annealing_step=5 * self.args.max_epoch)
                meta_loss = self.loss(alpha=meta_alpha, target=label_shot, epoch_num=epoch,
                                      num_classes=5, annealing_step=5 * self.args.max_epoch)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        # outer loop
        pre_logits_q = self.pre_base_learner(pre_embedding_query, pre_fast_weights)
        pre_evidence_q = softplus_evidence(pre_logits_q)
        meta_logits_q = self.meta_base_learner(meta_embedding_query, meta_fast_weights)
        meta_evidence_q = softplus_evidence(meta_logits_q)
        total_evidence_q = 10 * pre_evidence_q + 10 * meta_evidence_q
        return total_evidence_q

    def within_domain_forward(self, data_shot, label_shot, data_query):
        # embeddings
        pre_embedding_shot = self.pre_encoder(data_shot)
        meta_embedding_shot = self.encoder(data_shot)

        pre_embedding_query = self.pre_encoder(data_query)
        meta_embedding_query = self.encoder(data_query)

        # base-learner outputs
        meta_logit = self.meta_base_learner(meta_embedding_shot)
        pre_logit = self.pre_base_learner(pre_embedding_shot)

        # losses
        meta_loss = F.cross_entropy(meta_logit, label_shot)
        pre_loss = F.cross_entropy(pre_logit, label_shot)

        # grads
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())

        # fast_weights
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))

        for i in range(1, self.args.update_step):
            if i <= int(0.4 * self.args.update_step):
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_loss = F.cross_entropy(meta_logit, label_shot)
                pre_loss = F.cross_entropy(pre_logit, label_shot)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
            else:
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_evidence = softplus_evidence(meta_logit)
                meta_alpha = 10 * meta_evidence + 1
                pre_evidence = softplus_evidence(pre_logit)
                pre_alpha = 10 * pre_evidence + 1
                pre_loss = self.loss(alpha=pre_alpha, target=label_shot, epoch_num=200,
                                     num_classes=5, annealing_step=self.args.max_epoch)
                meta_loss = self.loss(alpha=meta_alpha, target=label_shot, epoch_num=200,
                                      num_classes=5, annealing_step=self.args.max_epoch)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        meta_logits_q = self.meta_base_learner(meta_embedding_query, meta_fast_weights)
        pre_logits_q = self.pre_base_learner(pre_embedding_query, pre_fast_weights)

        pre_evidence_q = softplus_evidence(pre_logits_q)
        meta_evidence_q = softplus_evidence(meta_logits_q)

        total_evidence_q = 10 * pre_evidence_q + 10 * meta_evidence_q

        return total_evidence_q

    def cross_domain_forward(self, data_shot, label_shot, data_query):
        # embeddings
        pre_embedding_shot = self.pre_encoder(data_shot)
        meta_embedding_shot = self.encoder(data_shot)

        pre_embedding_query = self.pre_encoder(data_query)
        meta_embedding_query = self.encoder(data_query)

        # base-learner outputs
        meta_logit = self.meta_base_learner(meta_embedding_shot)
        pre_logit = self.pre_base_learner(pre_embedding_shot)

        # losses
        meta_loss = F.cross_entropy(meta_logit, label_shot)
        pre_loss = F.cross_entropy(pre_logit, label_shot)

        # grads
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())

        # fast_weights
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))

        for i in range(1, self.args.update_step):
            if i <= int(0.4 * self.args.update_step):
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_loss = F.cross_entropy(meta_logit, label_shot)
                pre_loss = F.cross_entropy(pre_logit, label_shot)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
            else:
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_evidence = softplus_evidence(meta_logit)
                meta_alpha = 10 * meta_evidence + 1
                pre_evidence = softplus_evidence(pre_logit)
                pre_alpha = 10 * pre_evidence + 1
                pre_loss = self.loss(alpha=pre_alpha, target=label_shot, epoch_num=200,
                                     num_classes=5, annealing_step=self.args.max_epoch)
                meta_loss = self.loss(alpha=meta_alpha, target=label_shot, epoch_num=200,
                                      num_classes=5, annealing_step=self.args.max_epoch)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        meta_logits_q = self.meta_base_learner(meta_embedding_query, meta_fast_weights)
        pre_logits_q = self.pre_base_learner(pre_embedding_query, pre_fast_weights)

        pre_evidence_q = softplus_evidence(pre_logits_q)
        meta_evidence_q = softplus_evidence(meta_logits_q)

        total_evidence_q = 10 * pre_evidence_q + 10 * meta_evidence_q

        return total_evidence_q

    def aleatoric_forward(self, data_shot, label_shot, data_query):
        # embeddings
        meta_embedding_shot = self.encoder(data_shot)
        meta_embedding_query = self.encoder(data_query)
        # return meta_embedding_shot, meta_embedding_query
        pre_embedding_shot = self.pre_encoder(data_shot)
        pre_embedding_query = self.pre_encoder(data_query)
        # return pre_embedding_shot, pre_embedding_query, meta_embedding_shot, meta_embedding_query

        # base-learner outputs
        pre_logit = self.pre_base_learner(pre_embedding_shot)
        meta_logit = self.meta_base_learner(meta_embedding_shot)
        pre_evidence = softplus_evidence(pre_logit)
        meta_evidence = softplus_evidence(meta_logit)

        pre_evidence_weight = self.args.pretrain_evidence_weight
        meta_evidence_weight = self.args.meta_evidence_weight
        annealing_step = self.args.annealing_step
        update_step = self.args.update_step
        epoch_num = 0


        # pre_alpha = pre_evidence_weight * pre_evidence + 1
        # meta_alpha = meta_evidence_weight * meta_evidence + 1

        total_evidence = pre_evidence_weight * pre_evidence + meta_evidence_weight * meta_evidence
        total_alpha = total_evidence + 1
        # losses
        pre_loss = self.loss(alpha=total_alpha, target=label_shot, epoch_num=epoch_num,
                             num_classes=5, annealing_step=annealing_step)
        meta_loss = self.loss(alpha=total_alpha, target=label_shot, epoch_num=epoch_num,
                              num_classes=5, annealing_step=annealing_step)

        # # ANIL mode
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())

        # fast_weights, ANIL mode
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))

        # # inner loop, ANIL mode
        for i in range(1, update_step):
            meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
            pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
            meta_evidence = softplus_evidence(meta_logit)
            pre_evidence = softplus_evidence(pre_logit)
            # pre_alpha = pre_evidence_weight * pre_evidence + 1
            # meta_alpha = meta_evidence_weight * meta_evidence + 1
            total_evidence = pre_evidence_weight * pre_evidence + meta_evidence_weight * meta_evidence
            total_alpha = total_evidence + 1
            pre_loss = self.loss(alpha=total_alpha, target=label_shot, epoch_num=epoch_num,
                                 num_classes=5, annealing_step=annealing_step)
            meta_loss = self.loss(alpha=total_alpha, target=label_shot, epoch_num=epoch_num,
                                  num_classes=5, annealing_step=annealing_step)
            pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
            meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
            pre_fast_weights = list(
                map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
            meta_fast_weights = list(
                map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        # outer loop
        pre_logits_q = self.pre_base_learner(pre_embedding_query, pre_fast_weights)
        pre_evidence_q = softplus_evidence(pre_logits_q)
        meta_logits_q = self.meta_base_learner(meta_embedding_query, meta_fast_weights)
        meta_evidence_q = softplus_evidence(meta_logits_q)
        total_evidence_q = pre_evidence_weight * pre_evidence_q + meta_evidence_weight * meta_evidence_q
        return total_evidence_q

    def threshold_forward(self, data_shot, label_shot, data_query, epoch):
        # embeddings
        meta_embedding_shot = self.encoder(data_shot)
        meta_embedding_query = self.encoder(data_query)
        pre_embedding_shot = self.pre_encoder(data_shot)
        pre_embedding_query = self.pre_encoder(data_query)

        # base-learner outputs
        pre_logit = self.pre_base_learner(pre_embedding_shot)
        meta_logit = self.meta_base_learner(meta_embedding_shot)
        meta_evidence = softplus_evidence(meta_logit)
        pre_evidence = softplus_evidence(pre_logit)
        pre_alpha = 1 * pre_evidence + 1
        meta_alpha = 10 * meta_evidence + 1
        # total_evidence = 1*pre_evidence + 10*meta_evidence
        # total_alpha = total_evidence + 1

        # losses
        pre_loss = self.loss(alpha=pre_alpha, target=label_shot, epoch_num=epoch,
                             num_classes=5, annealing_step=self.args.max_epoch)
        meta_loss = self.loss(alpha=meta_alpha, target=label_shot, epoch_num=epoch,
                              num_classes=5, annealing_step=self.args.max_epoch)

        # ANIL mode
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())

        # fast_weights, ANIL mode
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))

        # inner loop, ANIL mode
        for i in range(1, self.args.update_step):
            meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
            pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
            meta_evidence = softplus_evidence(meta_logit)
            pre_evidence = softplus_evidence(pre_logit)
            pre_alpha = 1 * pre_evidence + 1
            meta_alpha = 10 * meta_evidence + 1
            # total_evidence = 1*pre_evidence + 10*meta_evidence
            # total_alpha = total_evidence + 1
            pre_loss = self.loss(alpha=pre_alpha, target=label_shot, epoch_num=epoch,
                                 num_classes=5, annealing_step=self.args.max_epoch)
            meta_loss = self.loss(alpha=meta_alpha, target=label_shot, epoch_num=epoch,
                                  num_classes=5, annealing_step=self.args.max_epoch)
            pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
            meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
            pre_fast_weights = list(
                map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
            meta_fast_weights = list(
                map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        # outer loop
        pre_logits_q = self.pre_base_learner(pre_embedding_query, pre_fast_weights)
        pre_evidence_q = softplus_evidence(pre_logits_q)
        meta_logits_q = self.meta_base_learner(meta_embedding_query, meta_fast_weights)
        meta_evidence_q = softplus_evidence(meta_logits_q)
        total_evidence_q = 1 * pre_evidence_q + 10 * meta_evidence_q
        return total_evidence_q


    def ood_forward(self, ID_data_shot, ID_label_shot, ID_data_query, OoD_data_query):
        # embeddings
        pre_ID_embedding_shot = self.pre_encoder(ID_data_shot)
        meta_ID_embedding_shot = self.encoder(ID_data_shot)

        pre_ID_embedding_query = self.pre_encoder(ID_data_query)
        meta_ID_embedding_query = self.encoder(ID_data_query)
        meta_OoD_embedding_query = self.encoder(OoD_data_query)

        # base-learner outputs
        meta_logit = self.meta_base_learner(meta_ID_embedding_shot)
        pre_logit = self.pre_base_learner(pre_ID_embedding_shot)

        # losses
        meta_loss = F.cross_entropy(meta_logit, ID_label_shot)
        pre_loss = F.cross_entropy(pre_logit, ID_label_shot)

        # grads
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())

        # fast_weights
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))

        for i in range(1, self.args.update_step):
            if i <= int(0.4 * self.args.update_step):
                meta_logit = self.meta_base_learner(meta_ID_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_ID_embedding_shot, pre_fast_weights)
                meta_loss = F.cross_entropy(meta_logit, ID_label_shot)
                pre_loss = F.cross_entropy(pre_logit, ID_label_shot)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
            else:
                meta_logit = self.meta_base_learner(meta_ID_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_ID_embedding_shot, pre_fast_weights)
                meta_evidence = softplus_evidence(meta_logit)
                meta_alpha = 10 * meta_evidence + 1
                pre_evidence = softplus_evidence(pre_logit)
                pre_alpha = 10 * pre_evidence + 1
                pre_loss = self.loss(alpha=pre_alpha, target=ID_label_shot, epoch_num=200,
                                     num_classes=5, annealing_step=self.args.max_epoch)
                meta_loss = self.loss(alpha=meta_alpha, target=ID_label_shot, epoch_num=200,
                                      num_classes=5, annealing_step=self.args.max_epoch)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        meta_ID_logits_q = self.meta_base_learner(meta_ID_embedding_query, meta_fast_weights)
        pre_ID_logits_q = self.pre_base_learner(pre_ID_embedding_query, pre_fast_weights)
        meta_OoD_logits_q = self.meta_base_learner(meta_OoD_embedding_query, meta_fast_weights)
        pre_ID_evidence_q = softplus_evidence(pre_ID_logits_q)
        meta_ID_evidence_q = softplus_evidence(meta_ID_logits_q)
        meta_OoD_evidence_q = softplus_evidence(meta_OoD_logits_q)
        total_ID_evidence_q = 10 * pre_ID_evidence_q + 10 * meta_ID_evidence_q
        total_OoD_evidence_q = meta_ID_evidence_q + meta_OoD_evidence_q
        return total_ID_evidence_q, total_OoD_evidence_q

    def active_forward(self, data_shot, label_shot, data_query, epoch):
        # embeddings
        meta_embedding_shot = self.encoder(data_shot)
        meta_embedding_query = self.encoder(data_query)
        pre_embedding_shot = self.pre_encoder(data_shot)
        pre_embedding_query = self.pre_encoder(data_query)

        # base-learner outputs
        meta_logit = self.meta_base_learner(meta_embedding_shot)
        pre_logit = self.pre_base_learner(pre_embedding_shot)

        # losses
        meta_loss = F.cross_entropy(meta_logit, label_shot)
        pre_loss = F.cross_entropy(pre_logit, label_shot)

        # grads
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())

        # fast_weights
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))

        # inner loop
        for i in range(1, 400):
            if i <= int(0.7 * self.args.update_step):
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_loss = F.cross_entropy(meta_logit, label_shot)
                pre_loss = F.cross_entropy(pre_logit, label_shot)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
            else:
                meta_logit = self.meta_base_learner(meta_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_embedding_shot, pre_fast_weights)
                meta_evidence = softplus_evidence(meta_logit)
                meta_alpha = 100 * meta_evidence + 1
                pre_evidence = softplus_evidence(pre_logit)
                pre_alpha = 1000 * pre_evidence + 1
                pre_loss = self.loss(alpha=pre_alpha, target=label_shot, epoch_num=0,
                                     num_classes=5, annealing_step=100)
                meta_loss = self.loss(alpha=meta_alpha, target=label_shot, epoch_num=0,
                                      num_classes=5, annealing_step=100)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        pre_logits_q = self.pre_base_learner(pre_embedding_query, pre_fast_weights)
        pre_evidence_q = softplus_evidence(pre_logits_q)
        meta_logits_q = self.meta_base_learner(meta_embedding_query, meta_fast_weights)
        meta_evidence_q = softplus_evidence(meta_logits_q)
        total_evidence_q = pre_evidence_q + 5 * meta_evidence_q
        return meta_logits_q, total_evidence_q, meta_fast_weights, pre_fast_weights


    def find_sample(self, ID_data_shot, ID_label_shot, ID_data_query, val_data_query, OoD_data_query):
        # embeddings
        pre_ID_embedding_shot = self.pre_encoder(ID_data_shot)
        meta_ID_embedding_shot = self.encoder(ID_data_shot)

        pre_ID_embedding_query = self.pre_encoder(ID_data_query)
        pre_OoD_embedding_query = self.pre_encoder(OoD_data_query)
        pre_val_embedding_query = self.pre_encoder(val_data_query)
        meta_ID_embedding_query = self.encoder(ID_data_query)
        meta_OoD_embedding_query = self.encoder(OoD_data_query)
        meta_val_embedding_query = self.encoder(val_data_query)

        # base-learner outputs
        meta_logit = self.meta_base_learner(meta_ID_embedding_shot)
        pre_logit = self.pre_base_learner(pre_ID_embedding_shot)

        # losses
        meta_loss = F.cross_entropy(meta_logit, ID_label_shot)
        pre_loss = F.cross_entropy(pre_logit, ID_label_shot)

        # grads
        meta_grad = torch.autograd.grad(meta_loss, self.meta_base_learner.parameters())
        pre_grad = torch.autograd.grad(pre_loss, self.pre_base_learner.parameters())

        # fast_weights
        meta_fast_weights = list(
            map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, self.meta_base_learner.parameters())))
        pre_fast_weights = list(
            map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, self.pre_base_learner.parameters())))

        for i in range(1, self.args.update_step):
            if i <= int(0.5 * self.args.update_step):
                meta_logit = self.meta_base_learner(meta_ID_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_ID_embedding_shot, pre_fast_weights)
                meta_loss = F.cross_entropy(meta_logit, ID_label_shot)
                pre_loss = F.cross_entropy(pre_logit, ID_label_shot)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
            else:
                meta_logit = self.meta_base_learner(meta_ID_embedding_shot, meta_fast_weights)
                pre_logit = self.pre_base_learner(pre_ID_embedding_shot, pre_fast_weights)
                meta_evidence = softplus_evidence(meta_logit)
                meta_alpha = meta_evidence + 1
                pre_evidence = softplus_evidence(pre_logit)
                pre_alpha = pre_evidence + 1
                pre_loss = self.loss(alpha=pre_alpha, target=ID_label_shot, epoch_num=200,
                                     num_classes=5, annealing_step=self.args.max_epoch)
                meta_loss = self.loss(alpha=meta_alpha, target=ID_label_shot, epoch_num=200,
                                      num_classes=5, annealing_step=self.args.max_epoch)
                pre_grad = torch.autograd.grad(pre_loss, pre_fast_weights)
                meta_grad = torch.autograd.grad(meta_loss, meta_fast_weights)
                pre_fast_weights = list(
                    map(lambda p: p[1] - self.pre_update_lr * p[0], zip(pre_grad, pre_fast_weights)))
                meta_fast_weights = list(
                    map(lambda p: p[1] - self.meta_update_lr * p[0], zip(meta_grad, meta_fast_weights)))
        meta_ID_logits_q = self.meta_base_learner(meta_ID_embedding_query, meta_fast_weights)
        pre_ID_logits_q = self.pre_base_learner(pre_ID_embedding_query, pre_fast_weights)
        pre_OoD_logits_q = self.pre_base_learner(pre_OoD_embedding_query, pre_fast_weights)
        meta_OoD_logits_q = self.meta_base_learner(meta_OoD_embedding_query, meta_fast_weights)
        pre_val_logits_q = self.pre_base_learner(pre_val_embedding_query, pre_fast_weights)
        meta_val_logits_q = self.meta_base_learner(meta_val_embedding_query, meta_fast_weights)
        pre_ID_evidence_q = softplus_evidence(pre_ID_logits_q)
        pre_OoD_evidence_q = softplus_evidence(pre_OoD_logits_q)
        pre_val_evidence_q = softplus_evidence(pre_val_logits_q)
        meta_val_evidence_q = softplus_evidence(meta_val_logits_q)
        meta_ID_evidence_q = softplus_evidence(meta_ID_logits_q)
        meta_OoD_evidence_q = softplus_evidence(meta_OoD_logits_q)
        total_ID_evidence_q = pre_ID_evidence_q + 6 * meta_ID_evidence_q
        total_val_evidence_q = pre_val_evidence_q + 3 * meta_val_evidence_q
        total_OoD_evidence_q = pre_OoD_evidence_q + meta_OoD_evidence_q
        return total_ID_evidence_q, total_val_evidence_q, total_OoD_evidence_q

