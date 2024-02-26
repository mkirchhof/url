""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import faiss
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_pred_correct(output, target):
    """Computes whether each target label is the top-1 prediction of the output"""
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return correct


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def recall_at_one(features, targets, mode="matmul"):
    if mode=="matmul":
        # Expects tensors as inputs
        features = F.normalize(features, dim=-1)
        closest_idxes = features.matmul(features.transpose(-2, -1)).topk(2)[1][:,1]
        closest_classes = targets[closest_idxes]
        is_same_class = (closest_classes == targets).float()
    elif mode=="faiss":
        # For big data, use faiss. Expects numpy arrays with float32 as inputs
        features = normalize(features, axis=1)
        faiss_search_index = faiss.IndexFlatIP(features.shape[-1])
        faiss_search_index.add(features)
        _, closest_idxes = faiss_search_index.search(features, 2)  # use 2, because the closest one will be the point itself
        closest_idxes = closest_idxes[:, 1]
        closest_classes = targets[closest_idxes]
        is_same_class = (closest_classes == targets).astype("float")
    else:
        raise NotImplementedError(f"mode {mode} not implemented.")

    return is_same_class.mean(), is_same_class

def recall_at_one_subset(features, targets, mode="matmul", idxes_database=None, idxes_query=None):
    if idxes_database is None:
        idxes_database = range(len(targets))
    idx_set_database = set(idxes_database)
    if idxes_query is None:
        idxes_query = range(len(targets))
    if mode=="matmul":
        # Expects tensors as inputs
        features = F.normalize(features, dim=-1)
        closest_idxes = features[idxes_query].matmul(features[idxes_database].transpose(-2, -1)).topk(2)[1][:,1]
        closest_classes = targets[idxes_database][closest_idxes]
        is_same_class = (closest_classes == targets[idxes_query]).float()
    elif mode=="faiss":
        # For big data, use faiss. Expects numpy arrays with float32 as inputs
        features = normalize(features, axis=1)
        faiss_search_index = faiss.IndexFlatIP(features.shape[-1])
        faiss_search_index.add(features[idxes_database])
        _, closest_idxes = faiss_search_index.search(features[idxes_query], 2)  # use 2, because the closest one will be the point itself
        closest_idxes = [closest_idxes[idx, 1] if idx in idx_set_database else closest_idxes[idx, 0] for idx in range(len(idxes_query))]
        closest_classes = targets[idxes_database][closest_idxes]
        is_same_class = (closest_classes == targets[idxes_query]).astype("float")
    else:
        raise NotImplementedError(f"mode {mode} not implemented.")

    return is_same_class.mean(), is_same_class

def pct_cropped_has_bigger_uncertainty(unc_orig, unc_cropped):
    return (unc_orig < unc_cropped).float().mean()


def save_image(img_tensor, title, path):
    plt.imshow(torch.minimum(torch.ones(1), torch.maximum(torch.zeros(1), img_tensor.permute(1, 2, 0) *
                                                          torch.tensor([0.2471, 0.2435, 0.2616]) + torch.tensor(
        [0.4914, 0.4822, 0.4465]))))
    plt.title(title)
    plt.savefig(path)
    plt.close()
