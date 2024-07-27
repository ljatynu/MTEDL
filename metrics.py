import torch
from sklearn import metrics


# Differential entropy for Dirichlet output
def compute_differential_entropy(log_alphas):
    alphas = torch.exp(log_alphas)
    alpha0 = torch.exp(torch.logsumexp(log_alphas, 1))
    loss = torch.sum(torch.lgamma(alphas), 1) - torch.lgamma(alpha0) - torch.sum(
        (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0).unsqueeze(-1)), 1)
    return loss


# Mutual Information for Dirichlet output
def compute_mutual_information(log_alphas):
    alphas = torch.exp(log_alphas)
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    alpha0 = torch.exp(log_alpha0)
    log_probs = log_alphas - log_alpha0.unsqueeze(-1)
    loss = -torch.sum(torch.exp(log_probs) * (log_probs -
                                              torch.digamma(alphas + 1) +
                                              torch.digamma(alpha0 + 1).unsqueeze(-1)),
                      1)
    return loss


# Precision for Dirichlet output
def compute_precision(log_alphas):
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    return torch.exp(log_alpha0)


def ROC_OOD(ood_Dent, ood_MI, ood_precision, all_label):
    # print('OOD Detection!')
    # Non-thresholded score is possible because the implementation only requires that the y_true can be sorted according to the y_score.
    auroc_Dent = metrics.roc_auc_score(all_label.numpy(), ood_Dent.numpy())
    auroc_MI = metrics.roc_auc_score(all_label.numpy(), ood_MI.numpy())
    auroc_precision = metrics.roc_auc_score(all_label.numpy(), -ood_precision.numpy())

    # print('AUROC score of Differential Entropy is', auroc_Dent)
    # print('AUROC score of Mutual Information is', auroc_MI)
    # print('AUROC score of precision is', auroc_precision)

    return auroc_Dent * 100, auroc_MI * 100, auroc_precision * 100