from pyro.infer import Predictive
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score, \
    balanced_accuracy_score
from torch.utils.data import random_split, DataLoader

from pems import load_model_bnn, PEMClass, SalientObstacleDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_roc(ax, fpr, tpr, auc, label_name: str):
    ax.plot(fpr, tpr, label=f"{label_name} ROC: {auc:.3f}")
    # ax.plot(np.linspace(0, 1), np.linspace(0, 1), color="black", linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()


def plot_prc(ax, rec, prec, avg_p, label_name: str):
    ax.plot(rec, prec, label=f"{label_name} Avg-P: {avg_p:.3f}")
    # ax_2.plot(np.linspace(0, 1), np.linspace(0, 1), color="blue", linestyle='--')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Prec")
    ax.legend()


def run():
    model = PEMClass(14, 1, use_cuda=True)
    guide = load_model_bnn("saved_models/pem_class_train_full_guide.pt",
                           "saved_models/pem_class_train_full_guide_params.pt")

    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)

    skip = 1
    full_data_set = SalientObstacleDataset(s_inputs[::skip], s_labels[::skip])
    combined_trus = []
    combined_preds = []

    b_size = 1024
    test_loader = DataLoader(full_data_set, batch_size=b_size, shuffle=True)
    test_predictive = Predictive(model, guide=guide, num_samples=800, return_sites={"obs", "_RETURN"})

    for eval_x, eval_y in test_loader:
        eval_x = eval_x.cuda()
        eval_y = eval_y.cuda()
        output = test_predictive(eval_x)
        pred_test_y = torch.sigmoid(output["_RETURN"]).mean(0)

        combined_preds.extend(pred_test_y.detach().cpu())
        combined_trus.extend(eval_y[:, 0].detach().cpu())

    fpr, tpr, ft_threshs = roc_curve(combined_trus, combined_preds)
    prec, rec, pr_threshs = precision_recall_curve(combined_trus, combined_preds)

    auc = roc_auc_score(combined_trus, combined_preds)
    avg_p = average_precision_score(combined_trus, combined_preds)
    accuracy = accuracy_score(combined_trus, np.round(combined_preds))
    b_accuracy = balanced_accuracy_score(combined_trus, np.round(combined_preds))

    print(f"AUC_ROC: {auc:.4f}, Accuracy: {accuracy}, B-Accuracy: {b_accuracy}, Average Precision: {avg_p}")

    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_roc(ax_1, fpr, tpr, auc, "Full Train")
    plot_prc(ax_2, rec, prec, avg_p, "Full Train")
    plt.show()


if __name__ == "__main__":
    run()
