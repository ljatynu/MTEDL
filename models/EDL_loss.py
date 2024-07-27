import torch
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    return F.one_hot(labels, num_classes)

def relu_evidence(logit):
    return F.relu(logit)


def exp_evidence(logit):
    return torch.exp(torch.clamp(logit, -10, 10))


def softplus_evidence(logit):
    return F.softplus(logit)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(one_hot_target, alpha, device=None):
    if not device:
        device = get_device()
    one_hot_target = one_hot_target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((one_hot_target - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(one_hot_target, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    one_hot_target = one_hot_target.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(one_hot_target, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - one_hot_target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, one_hot_target, alpha, epoch_num, num_classes, annealing_step, device=None):
    one_hot_target = one_hot_target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(one_hot_target * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - one_hot_target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    one_hot_target = one_hot_embedding(target, num_classes)
    loss = torch.mean(
        mse_loss(one_hot_target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    one_hot_target = one_hot_embedding(target, num_classes)
    loss = torch.mean(
        edl_loss(
            torch.log, one_hot_target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    alpha, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    one_hot_target = one_hot_embedding(target, num_classes)
    loss = torch.mean(
        edl_loss(
            torch.digamma, one_hot_target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
