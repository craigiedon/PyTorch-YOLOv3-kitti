from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, roc_curve, \
    precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from eval_full_tp_PEM import plot_roc, plot_prc
from pems import SalientObstacleDataset


def logistic_baseline(training_set, test_set):
    training_ins, training_labels = training_set[:][0], training_set[:][1][:, 0]
    test_ins, test_labels = test_set[:][0], test_set[:][1][:, 0]

    lr = LogisticRegression(max_iter=10000)
    lr.fit(training_ins, training_labels)

    lr_preds = lr.predict(test_ins)
    lr_probs = lr.predict_proba(test_ins)[:, 1]

    fpr, tpr, ft_threshs = roc_curve(test_labels, lr_probs)
    prec, rec, pr_threshs = precision_recall_curve(test_labels, lr_probs)

    auc = roc_auc_score(test_labels, lr_probs)
    avg_p = average_precision_score(test_labels, lr_probs)
    accuracy = accuracy_score(test_labels, lr_preds)
    b_accuracy = balanced_accuracy_score(test_labels, lr_preds)

    bce = F.binary_cross_entropy(torch.tensor(lr_probs, dtype=torch.float), test_labels)

    metrics = {"roc_curve": (fpr, tpr, ft_threshs),
               "pr_curve": (prec, rec, pr_threshs),
               "auc": auc,
               "avg_p": avg_p,
               "accuracy": accuracy,
               "b_accuracy": b_accuracy,
               "bce": bce
               }

    return lr, metrics


def run():
    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"
    s_inputs = torch.tensor(np.loadtxt(salient_input_path), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt(salient_label_path), dtype=torch.float)

    full_data_set = SalientObstacleDataset(s_inputs, s_labels)

    fig, (ax_1, ax_2) = plt.subplots(1, 2)
    kf = StratifiedKFold(n_splits=10)
    for ki, (train_idx, test_idx) in enumerate(kf.split(full_data_set, full_data_set.s_label[:, 0])):
        training_set = Subset(full_data_set, train_idx)
        test_set = Subset(full_data_set, test_idx)

        lr, metrics = logistic_baseline(training_set, test_set)
        print(f"K: {ki} --- AUC: {metrics['auc']}, Avg-P: {metrics['avg_p']}, Accuracy: {metrics['accuracy']}, b_accuracy: {metrics['b_accuracy']}")

        fpr, tpr, _ = metrics["roc_curve"]
        prec, rec, _ = metrics["pr_curve"]

        plot_roc(ax_1, fpr, tpr, metrics['auc'], f"K-{ki}")
        plot_prc(ax_2, rec, prec, metrics['avg_p'], f"K-{ki}")

    plt.show()


if __name__ == "__main__":
    run()