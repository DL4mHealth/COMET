import numpy as np
import sklearn
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch


def eval_classification(model, train_data, train_labels, test_data, test_labels, fraction=None):
    """
    Args:
      fraction (Union[float, NoneType]): The fraction of training data. It used to do semi-supervised learning.
    """

    assert train_labels.ndim == 1 or train_labels.ndim == 2

    if fraction:
        # use first fraction number of training data
        train_data = train_data[:int(train_data.shape[0]*fraction)]
        train_labels = train_labels[:int(train_labels.shape[0]*fraction)]
        # print(f"Fraction of train data used for semi_supervised learning:{fraction}\n")

    train_repr = model.encode(train_data)
    test_repr = model.encode(test_data)

    fit_clf = eval_protocols.fit_lr
    clf = fit_clf(train_repr, train_labels)

    pred_prob = clf.predict_proba(test_repr)
    # print(pred_prob.shape)
    target_prob = (F.one_hot(torch.tensor(test_labels).long(), num_classes=int(train_labels.max()+1))).numpy()
    # print(target_prob.shape)
    pred = pred_prob.argmax(axis=1)
    target = test_labels

    metrics_dict = {}
    metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
    metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, average='macro', multi_class='ovr')
    metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob, average='macro')

    return metrics_dict
