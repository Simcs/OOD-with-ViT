import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
)


def auroc(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    return fpr, tpr, auc(fpr, tpr)


def aupr(labels, preds):
    precision, recall, _ = precision_recall_curve(labels, preds)
    return precision, recall, auc(recall, precision)


def fpr_at_95_tpr(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    
    if all(tpr < 0.95):
        # no threshold allows tpr >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # all thresholds allow tpr >= 0.95, so find lowest possible fpr
        indices = [i for i, x in enumerate(tpr) if x >= 0.95]
        fpr_at_95_tprs = fpr[indices]
        return min(fpr_at_95_tprs)
    else:
        # linear interpolation between values to get fpr at tpr == 0.95
        return np.interp(0.95, tpr, fpr) 