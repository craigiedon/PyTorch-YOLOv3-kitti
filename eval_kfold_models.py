from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from eval_full_tp_PEM import plot_roc, plot_prc
from full_tp_pem import eval_model
from logistic_baseline import logistic_baseline
from pems import load_model_bnn, PEMClass, SalientObstacleDataset, load_model_det, PEMClass_Deterministic
from os.path import join
import matplotlib.pyplot as plt
import torch
import numpy as np

from train_tp_pem_det import eval_model_det


def average_curves(xyt_rollouts):
    interp_curves = []
    for (x, y, t) in xyt_rollouts:
        interp_c = np.interp(np.linspace(0, 1, 1000), x, y)
        interp_curves.append(interp_c)

    mu_curve = np.mean(interp_curves, axis=0)
    std_curve = np.std(interp_curves, axis=0)
    return mu_curve, std_curve


def avg_metrics(metrics):
    avg_met = {}
    for k, v in metrics.items():
        if "curve" in k:
            avg_met[k] = average_curves(v)
        else:
            avg_met[k] = np.mean(v, axis=0)

    return avg_met


def eval_kfold_models(dataset, k):
    kf = StratifiedKFold(n_splits=k)

    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 5))

    bnn_metrics = defaultdict(list)
    lr_metrics = defaultdict(list)
    det_metrics = defaultdict(list)

    for ki, (train_idx, test_idx) in enumerate(kf.split(dataset, dataset.s_label[:, 0])):
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        bnn_results_folder = f"saved_models/bnn_k{ki}"
        kth_model = PEMClass(14, 1, use_cuda=True)
        kth_guide = load_model_bnn(join(bnn_results_folder, "pem_class_train_full_guide.pt"),
                                   join(bnn_results_folder, "pem_class_train_full_guide_params.pt"))

        det_model = load_model_det(PEMClass_Deterministic(14, 1, use_cuda=True), f"saved_models/det_baseline_k{ki}/pem_class_train_full")

        kth_metrics = eval_model(kth_model, kth_guide, test_set)
        lr, kth_lr_metric = logistic_baseline(train_set, test_set)
        kth_det_metric = eval_model_det(det_model, test_set)

        for k, v in kth_metrics.items():
            bnn_metrics[k].append(v)

        for k, v in kth_lr_metric.items():
            lr_metrics[k].append(v)

        for k, v in kth_det_metric.items():
            det_metrics[k].append(v)

        print(f"K-{ki}:")
        for baseline_metric in [kth_metrics, kth_lr_metric, kth_det_metric]:
            print(f"\tBCE: {baseline_metric['bce']} Accuracy: {baseline_metric['accuracy']}, AUC-ROC: {baseline_metric['auc']}, Avg-Precision: {baseline_metric['avg_p']}, Lowest-Prob: {baseline_metric['lowest_prob']}")

    avg_bnn_metrics = avg_metrics(bnn_metrics)
    avg_lr_metrics = avg_metrics(lr_metrics)
    avg_det_metrics = avg_metrics(det_metrics)

    print(f"BNN BCE: {avg_bnn_metrics['bce']}, LR BCE: {avg_lr_metrics['bce']} Det BCE: {avg_det_metrics['bce']}")

    plot_roc(ax_1, np.linspace(0, 1, 1000), avg_bnn_metrics['roc_curve'][0], avg_bnn_metrics['auc'], "BNN")
    ax_1.fill_between(np.linspace(0, 1, 1000),
                      avg_bnn_metrics['roc_curve'][0] - avg_bnn_metrics['roc_curve'][1],
                      avg_bnn_metrics['roc_curve'][0] + avg_bnn_metrics['roc_curve'][1], alpha=0.15)

    plot_roc(ax_1, np.linspace(0, 1, 1000), avg_lr_metrics['roc_curve'][0], avg_lr_metrics['auc'], "LR")
    plot_roc(ax_1, np.linspace(0, 1, 1000), avg_det_metrics['roc_curve'][0], avg_det_metrics['auc'], "Det")

    plot_prc(ax_2, np.linspace(0, 1, 1000), avg_bnn_metrics['pr_curve'][0], avg_bnn_metrics['avg_p'], "BNN")
    ax_2.fill_between(np.linspace(0, 1, 1000),
                      avg_bnn_metrics['pr_curve'][0] - avg_bnn_metrics['pr_curve'][1],
                      avg_bnn_metrics['pr_curve'][0] + avg_bnn_metrics['pr_curve'][1], alpha=0.15)
    ylims = ax_2.get_ylim()

    plot_prc(ax_2, np.linspace(0, 1, 1000), avg_lr_metrics['pr_curve'][0], avg_lr_metrics['avg_p'], "LR")
    plot_prc(ax_2, np.linspace(0, 1, 1000), avg_det_metrics['pr_curve'][0], avg_det_metrics['avg_p'], "Det")

    ax_2.set_ylim(ylims)

    plt.show()


if __name__ == "__main__":
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    eval_kfold_models(full_data_set, 5)
