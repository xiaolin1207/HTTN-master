from __future__ import print_function, absolute_import
from sklearn import metrics
import numpy as np
# __all__ = ['accuracy']
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    microF1= metrics.f1_score(y_true, y_pred, average="micro")
    macroF1 = metrics.f1_score(y_true, y_pred, average="macro")
    return microF1,macroF1
def calc_acc(y_true, y_pred):
    # y_pred[y_pred > 0.5] = 1
    # y_pred[y_pred <= 0.5] = 0
    return metrics.accuracy_score(y_true, y_pred)


def precision_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]

    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean()
        precision.append(p * 100 / batch_size)

    return precision


def ndcg_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]

    ndcg = []
    for _k in k:
        score = 0
        rank = np.log2(np.arange(2, 2 + _k))
        for i in range(batch_size):
            l = label[i, pred[i, :_k]]
            n = l.sum()
            if (n == 0):
                continue

            dcg = (l / rank).sum()
            label_count = label[i].sum()
            norm = 1 / np.log2(np.arange(2, 2 + np.min((_k, label_count))))
            norm = norm.sum()
            score += dcg / norm

        ndcg.append(score * 100 / batch_size)
    return ndcg