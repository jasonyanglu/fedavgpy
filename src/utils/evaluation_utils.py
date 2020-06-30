import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score


def extended_gmean(y, pred, unique_class):

    result = 1
    for j in unique_class:
        tp = np.sum(np.logical_and(pred == j, y == j))
        p = np.sum(y == j)
        result *= tp / p

    return np.power(result, 1 / len(unique_class))


def macro_avg_acc(y, pred, unique_class):

    result = []
    y = np.array(y)
    pred = np.array(pred)
    for j in unique_class:
        tp = np.sum(np.logical_and(pred == j, y == j))
        p = np.sum(y == j)
        result.append(tp / p)

    return np.mean(result)


def macro_f1_score(y, pred, unique_class):

    pred_unique_class = np.unique(pred)
    f1 = f1_score(y, pred, average=None)

    result = []
    for j in unique_class:
        if j in pred_unique_class:
            idx = list(pred_unique_class).index(j)
            result.append(f1[idx])
        else:
            result.append(0)

    return np.mean(result)


def evaluate_multiclass(y, pred):

    unique_class = np.unique(y)

    return_dict = {}
    return_dict['mauc'] = roc_auc_score(y, pred, average='macro', multi_class='ovo')
    return_dict['egmean'] = extended_gmean(y, pred, unique_class)
    return_dict['mfm'] = macro_f1_score(y, pred, unique_class)
    return_dict['mava'] = macro_avg_acc(y, pred, unique_class)
    return_dict['acc'] = accuracy_score(y, pred)

    return return_dict
