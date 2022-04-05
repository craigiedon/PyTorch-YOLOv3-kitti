from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from eval_full_tp_PEM import plot_roc, plot_prc
from full_tp_pem import eval_model
from logistic_baseline import logistic_baseline
from pems import load_model, PEMClass, SalientObstacleDataset
from os.path import join
import matplotlib.pyplot as plt
import torch
import numpy as np


def eval_kfold_models(dataset, k):
    kf = StratifiedKFold(n_splits=k)

    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 5))

    pr_curves = []
    roc_curves = []
    aucs = []
    avg_ps = []
    bces = []

    lr_pr_curves = []
    lr_roc_curves = []
    lr_aucs = []
    lr_avg_ps = []
    lr_bces = []

    for ki, (train_idx, test_idx) in enumerate(kf.split(dataset, dataset.s_label[:, 0])):
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        results_folder = f"saved_models/pem_class_k{ki}"
        kth_model = PEMClass(14, 2, use_cuda=True)
        kth_guide = load_model(join(results_folder, "pem_class_train_full_guide.pt"),
                               join(results_folder, "pem_class_train_full_guide_params.pt"))
        eval_metrics = eval_model(kth_model, kth_guide, test_set)

        fpr, tpr, _ = eval_metrics["roc_curve"]
        prec, rec, _ = eval_metrics["pr_curve"]

        pr_curves.append(np.interp(np.linspace(0, 1, 1000), list(reversed(rec)), list(reversed(prec))))
        roc_curves.append(np.interp(np.linspace(0, 1, 1000), fpr, tpr))

        lr, lr_metrics = logistic_baseline(train_set, test_set)

        lr_fpr, lr_tpr, _ = lr_metrics["roc_curve"]
        lr_prec, lr_rec, _ = lr_metrics["pr_curve"]
        lr_pr_curves.append(np.interp(np.linspace(0, 1, 1000), list(reversed(lr_rec)), list(reversed(lr_prec))))
        lr_roc_curves.append(np.interp(np.linspace(0, 1, 1000), lr_fpr, lr_tpr))

        aucs.append(eval_metrics["auc"])
        avg_ps.append(eval_metrics["avg_p"])
        accuracy = eval_metrics["accuracy"]
        bces.append(eval_metrics["bce"])

        lr_aucs.append(lr_metrics["auc"])
        lr_avg_ps.append(lr_metrics["avg_p"])
        lr_bces.append(lr_metrics["bce"])

        print(
            f"K-{ki}\t BCE: {eval_metrics['bce']} Accuracy: {accuracy}, AUC-ROC: {aucs[-1]}, Avg-Precision: {avg_ps[-1]}")

        # plot_roc(ax_1, fpr, tpr, auc, f"Fold-{ki}")
        # plot_prc(ax_2, np.linspace(0, 1, 1000), interp_pr_curve, avg_p, f"Fold-{ki}")

    print(f"BNN BCE: {np.mean(bces)}, LR BCE: {np.mean(lr_bces)}")
    mu_roc, std_roc = np.mean(roc_curves, axis=0), np.std(roc_curves, axis=0)
    mu_prc, std_prc = np.mean(pr_curves, axis=0), np.std(pr_curves, axis=0)

    lr_mu_roc, lr_std_roc = np.mean(lr_roc_curves, axis=0), np.std(lr_roc_curves, axis=0)
    lr_mu_prc, lr_std_prc = np.mean(lr_pr_curves, axis=0), np.std(lr_pr_curves, axis=0)

    plot_roc(ax_1, np.linspace(0, 1, 1000), mu_roc, np.mean(aucs), "BNN")
    ax_1.fill_between(np.linspace(0, 1, 1000), mu_roc - std_roc, mu_roc + std_roc, alpha=0.15)

    plot_roc(ax_1, np.linspace(0, 1, 1000), lr_mu_roc, np.mean(lr_aucs), "LR")

    plot_prc(ax_2, np.linspace(0, 1, 1000), mu_prc, np.mean(avg_ps), "BNN")
    ylims = ax_2.get_ylim()
    ax_2.fill_between(np.linspace(0, 1, 1000), mu_prc - std_prc, mu_prc + std_prc, alpha=0.15)

    plot_prc(ax_2, np.linspace(0, 1, 1000), lr_mu_prc, np.mean(lr_avg_ps), "LR")

    ax_2.set_ylim(ylims)

    plt.show()


if __name__ == "__main__":
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    eval_kfold_models(full_data_set, 10)
