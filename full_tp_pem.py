from pems import PEMClass, save_model_bnn
import pyro
import time
import numpy as np
import os
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, precision_recall_curve, \
    average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from os.path import join

from matplotlib import pyplot as plt
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI, Predictive, TraceMeanField_ELBO
import pyro
import pyro.distributions as dist
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch
import torch.nn.functional as F

from load_salient_dataset import group_data, count_tru_pos_v_false_neg, discretize_truncation, discretize_trunc_tensor, \
    filter_inp_labels
from pems import PEMClass, SalientObstacleDataset
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal

import numpy as np
import torch


def k_fold_validaton_bnn(dataset, k, train_func, experiment_name: str):
    kf = StratifiedKFold(n_splits=k)
    print(len(dataset))
    print(len(dataset.s_label[:, 0]))

    for ki, (train_idx, test_idx) in enumerate(kf.split(dataset, dataset.s_label[:, 0])):
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        results_folder = f"saved_models/{experiment_name}_k{ki}"
        os.makedirs(results_folder, exist_ok=True)
        kth_model, kth_guide = train_func(train_set, 1000, results_folder)
        eval_metrics = eval_model(kth_model, kth_guide, test_set)
        print(f"K-{ki}\t Accuracy: {eval_metrics['accuracy']}, AUC-ROC: {eval_metrics['auc']}, Avg-Precision: {eval_metrics['avg_p']}")


def train_model(train_set: Dataset, epochs: int, results_folder: str):
    print(results_folder)
    print(f"Training Set Size: {len(train_set)}")

    b_size = 8000
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)

    model = PEMClass(14, 1, 20, use_cuda=True)
    pyro.clear_param_store()

    guide = AutoLowRankMultivariateNormal(poutine.block(model, hide=['obs'])).cuda()

    adam = pyro.optim.Adam({"lr": 1e-3})
    loss_elb = Trace_ELBO(vectorize_particles=True)
    svi = SVI(model, guide, adam, loss=loss_elb)

    # Training Step
    for j in range(epochs):
        train_loss = 0
        for train_x, train_y in train_loader:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            train_loss += svi.step(train_x, train_y[:, [0]])

        train_loss = train_loss / len(train_set)

        if j % 10 == 0:
            print(f"[t:  {j + 1}] ELBO: {train_loss:.4f}")

    save_model_bnn(guide, pyro.get_param_store(), join(results_folder, "pem_class_train_full"))
    return model, guide


def eval_model(model, guide, test_set):
    combined_preds = torch.tensor([]).cuda()
    combined_trus = torch.tensor([]).cuda()

    test_loader = DataLoader(test_set, 8000)
    test_predictive = Predictive(model, guide=guide, num_samples=800, return_sites={"obs", "_RETURN"})

    for test_x, test_y in test_loader:
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        pred_test_y = torch.sigmoid(test_predictive(test_x)["_RETURN"]).mean(0)

        combined_preds = torch.cat((combined_preds, pred_test_y))
        combined_trus = torch.cat((combined_trus, test_y[:, [0]]))

    combined_preds = combined_preds.detach().cpu().numpy()
    combined_trus = combined_trus.detach().cpu().numpy()

    fpr, tpr, ft_threshs = roc_curve(combined_trus, combined_preds)
    prec, rec, pr_threshs = precision_recall_curve(combined_trus, combined_preds)

    bce = F.binary_cross_entropy(torch.tensor(combined_preds), torch.tensor(combined_trus)).item()

    auc = roc_auc_score(combined_trus, combined_preds)
    avg_p = average_precision_score(combined_trus, combined_preds)
    accuracy = accuracy_score(combined_trus, np.round(combined_preds))
    b_accuracy = balanced_accuracy_score(combined_trus, np.round(combined_preds))

    return {"roc_curve": (fpr, tpr, ft_threshs),
            "pr_curve": (list(reversed(rec)), list(reversed(prec)), pr_threshs),
            "auc": auc,
            "avg_p": avg_p,
            "accuracy": accuracy,
            "b_accuracy": b_accuracy,
            "bce": bce,
            "lowest_prob": np.min(combined_preds)
            }


def run():
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    # k_fold_validaton_bnn(full_data_set, 5, train_model, "bnn")
    train_model(full_data_set, 1000, "saved_models/bnn_full/")


"""
def test_run():
    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"
    s_inputs = torch.tensor(np.loadtxt(salient_input_path), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt(salient_label_path), dtype=torch.float)

    skip = 1
    full_data_set = SalientObstacleDataset(s_inputs[::skip], s_labels[::skip])
    # Train 80%, Test 20%?
    train_size = int(len(full_data_set) * 0.95)
    test_size = len(full_data_set) - train_size

    print(train_size)
    print(test_size)
    train_set, test_set = random_split(full_data_set, [train_size, test_size],
                                       generator=torch.Generator().manual_seed(42))

    # dets = full_data_set[:][1][:, 0]
    # fns_mask = dets == 0
    # fns = torch.masked_select(dets, fns_mask)
    # print(f"Num False Negatives: {len(fns)} / {len(full_data_set)}")
    # print(f"Accuracy of guessing all 1s: {(len(full_data_set) - len(fns)) / len(full_data_set)}")

    b_size = 8000
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=b_size, shuffle=True)

    num_ins = len(train_set[0][0])
    assert num_ins == 14

    model = PEMClass(num_ins, 2, 20, use_cuda=True)
    pyro.clear_param_store()

    num_iterations = 10000
    guide = AutoDiagonalNormal(poutine.block(model, hide=['obs'])).cuda()
    p_store = pyro.get_param_store()

    adam = pyro.optim.Adam({"lr": 1e-3})
    loss_elb = Trace_ELBO(vectorize_particles=True)
    svi = SVI(model, guide, adam, loss=loss_elb)

    training_losses = []
    test_losses = []

    # Training Step
    for j in range(num_iterations):
        train_loss = 0
        for train_x, train_y in train_loader:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            train_loss += svi.step(train_x, train_y[:, 0])

        train_loss = train_loss / train_size

        if j % 10 == 0:
            print(f"[t:  {j + 1}] ELBO: {train_loss:.4f}")

        if j % 250 == 0:
            # Evaluation Step
            test_predictive = Predictive(model, guide=guide, num_samples=250, return_sites={"obs", "_RETURN"})

            metrics = []
            combined_preds = []
            combined_trus = []

            for eval_x, eval_y in train_loader:
                eval_x = eval_x.cuda()
                eval_y = eval_y.cuda()
                pred_test_y = F.softmax(test_predictive(eval_x)["_RETURN"], dim=2).mean(0)[:, 1]

                combined_preds.extend(pred_test_y.detach().cpu())
                combined_trus.extend(eval_y[:, 0].detach().cpu())

            fpr, tpr, ft_threshs = roc_curve(combined_trus, combined_preds)
            prec, rec, pr_threshs = precision_recall_curve(combined_trus, combined_preds)

            auc = roc_auc_score(combined_trus, combined_preds)
            avg_p = average_precision_score(combined_trus, combined_preds)
            accuracy = accuracy_score(combined_trus, np.round(combined_preds))
            b_accuracy = balanced_accuracy_score(combined_trus, np.round(combined_preds))

            training_losses.append(train_loss)
            print(
                f"[t:  {j + 1}] ELBO: {train_loss:.4f} AUC_ROC: {auc:.4f}, Accuracy: {accuracy}, B-Accuracy: {b_accuracy}, Average Precision: {avg_p}")

            save_model_bnn(guide, pyro.get_param_store(), f"saved_models/checkpoints/pem_class_train_e{j}")

    save_model_bnn(guide, pyro.get_param_store(), "saved_models/pem_class_train_full")

    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 5))
    ax_1.plot(fpr, tpr, label=f"ROC: {auc:.4f}", color="orange")
    ax_1.plot(np.linspace(0, 1), np.linspace(0, 1), color="blue", linestyle='--')
    ax_1.set_xlabel("False Positive Rate")
    ax_1.set_ylabel("True Positive Rate")
    ax_1.legend()

    ax_2.plot(rec, prec, label=f"Average P: {avg_p:.4f}", color="orange")
    # ax_2.plot(np.linspace(0, 1), np.linspace(0, 1), color="blue", linestyle='--')
    ax_2.set_xlabel("Recall")
    ax_2.set_ylabel("Prec")
    ax_2.legend()

    plt.show()
"""


if __name__ == "__main__":
    run()
