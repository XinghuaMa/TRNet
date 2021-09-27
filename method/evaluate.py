from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd

import features as feat
from config import opt


def binary_confusion_matrix(pred, target):
    pre_neg = (pred == 0)
    tar_neg = (target == 0)
    TN = np.sum(np.logical_and(pre_neg, tar_neg) * 1)
    TP = np.sum(np.logical_and(pred, target) * 1)
    FN = np.sum(pre_neg) - TN
    FP = np.sum(pred) - TP
    return TN, TP, FN, FP


def calculate_metrics(TN, TP, FN, FP):
    ACC = 0 if (TN + TP + FN + FP) == 0 else (TN + TP) / (TN + TP + FN + FP)
    Sens = 0 if (TP + FN) == 0 else TP / (TP + FN)
    Spec = 0 if (TN + FP) == 0 else TN / (TN + FP)
    PPV = 0 if (TP + FP) == 0 else TP / (TP + FP)
    NPV = 0 if (TN + FN) == 0 else TN / (TN + FN)
    F1 = 0 if (2 * TP + FP + FN) == 0 else 2 * TP / (2 * TP + FP + FN)
    MCC = 0 if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) == 0 else (TP * TN - FP * FN) / (
            ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    return ACC, Sens, Spec, PPV, NPV, F1, MCC


def get_index(pred, target, num_index=7):
    pred = torch.argmax(pred, dim=1)
    if opt.use_gpu:
        pred, target = pred.cpu(), target.cpu()
    pred, target = pred.detach().numpy(), target.detach().numpy()

    TN, TP, FN, FP = binary_confusion_matrix(pred, target)
    ACC, Sens, Spec, PPV, NPV, F1, MCC = calculate_metrics(TN, TP, FN, FP)
    return [ACC, Sens, Spec, PPV, NPV, F1, MCC][:num_index]


def evaluate(model, eval_dataLoader, num_indexes=7):
    index_counter = []
    for i in range(num_indexes):
        index_counter += [feat.Counter()]

    for batch_id, (sequence_image, sequence_label) in enumerate(eval_dataLoader):
        Input, target = sequence_image.requires_grad_(), sequence_label

        if opt.use_gpu:
            Input, target = Input.cuda(), target.cuda()

        pred = model(Input)
        pred, target = pred.view(-1, 2), target.view(-1)

        indexes = get_index(pred, target, num_index=num_indexes)

        for j in range(num_indexes):
            index_counter[j].updata(indexes[j])

    ret_index = []
    for i in range(num_indexes):
        ret_index += [index_counter[i].avg]
    return ret_index


def print_evaluate_index(model, eval_dataLoader, num_indexes=7):
    name_index = ['ACC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'MCC']
    ret_index = evaluate(model, eval_dataLoader, num_indexes=7)
    for i in range(num_indexes):
        print(f'{name_index[i]}:{round(ret_index[i] * 100, 2)}', end=' ')
    print('\n')
    return
