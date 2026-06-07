"""
Metrics: accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrix.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix, classification_report,
)


@torch.no_grad()
def compute_metrics(y_true, y_pred, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
        labels=list(range(len(class_names))) if class_names else None,
    )

    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(class_names))) if class_names else None,
    )

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4, zero_division=0,
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_precision": p,
        "per_class_recall": r,
        "per_class_f1": f,
        "per_class_support": s,
        "confusion_matrix": cm,
        "classification_report": report,
    }


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
