from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from eval_full_tp_PEM import plot_roc, plot_prc
from full_tp_pem import eval_model
from detection_baselines import logistic_baseline, guess_one, guess_mu
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

    baseline_metrics = {
        "bnn": defaultdict(list),
        "lr": defaultdict(list),
        "det": defaultdict(list),
        # "g_1": defaultdict(list),
        "g_mu": defaultdict(list)
    }

    for ki, (train_idx, test_idx) in enumerate(kf.split(dataset, dataset.s_label[:, 0])):
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        bnn_results_folder = f"saved_models/bnn_k{ki}"
        kth_model = PEMClass(14, 1, use_cuda=True)
        kth_guide = load_model_bnn(join(bnn_results_folder, "pem_class_train_full_guide.pt"),
                                   join(bnn_results_folder, "pem_class_train_full_guide_params.pt"))

        det_model = load_model_det(PEMClass_Deterministic(14, 1, use_cuda=True),
                                   f"saved_models/det_baseline_k{ki}/pem_class_train_full")

        kth_metrics = eval_model(kth_model, kth_guide, test_set)
        lr, kth_lr_metric = logistic_baseline(train_set, test_set)
        kth_det_metric = eval_model_det(det_model, test_set)
        # kth_g1_metric = guess_one(test_set)
        kth_mu_metric = guess_mu(train_set, test_set)

        for k, v in kth_metrics.items():
            baseline_metrics["bnn"][k].append(v)

        for k, v in kth_lr_metric.items():
            baseline_metrics["lr"][k].append(v)

        for k, v in kth_det_metric.items():
            baseline_metrics["det"][k].append(v)

        # for k, v in kth_g1_metric.items():
        #     baseline_metrics["g_1"][k].append(v)

        for k, v in kth_mu_metric.items():
            baseline_metrics["g_mu"][k].append(v)

    avg_b_metrics = {k: avg_metrics(v) for k, v in baseline_metrics.items()}

    print(f"BCEs: ")
    for k, v in avg_b_metrics.items():
        print(f"\t{k}: {v['bce']:.3f}")

    print("Acc:")
    for k, v in avg_b_metrics.items():
        print(f"\t{k}: {v['accuracy']:.3f}")

    _, ax_1 = plt.subplots(figsize=(5, 5))
    for k, v in avg_b_metrics.items():
        plot_roc(ax_1, np.linspace(0, 1, 1000), v['roc_curve'][0], v['auc'], k)
        ax_1.fill_between(np.linspace(0, 1, 1000),
                          v['roc_curve'][0] - v['roc_curve'][1],
                          v['roc_curve'][0] + v['roc_curve'][1], alpha=0.15)

    _, ax_2 = plt.subplots(figsize=(5, 5))
    for k, v in avg_b_metrics.items():
        plot_prc(ax_2, np.linspace(0, 1, 1000), v['pr_curve'][0], v['auc'], k)
        ax_2.fill_between(np.linspace(0, 1, 1000),
                          v['pr_curve'][0] - v['pr_curve'][1],
                          v['pr_curve'][0] + v['pr_curve'][1], alpha=0.15)

    # ylims = ax_2.get_ylim()
    # ax_2.set_ylim(ylims)

    plt.show()


if __name__ == "__main__":
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    eval_kfold_models(full_data_set, 5)
