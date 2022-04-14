import os
from os.path import join

import torch
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score, \
    balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from torch import optim
import torch.nn.functional as F

from full_tp_pem import k_fold_validaton_bnn
from pems import SalientObstacleDataset, PEMClass_Deterministic, save_model_bnn, save_model_det


def run():
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)

    k_fold_validaton_det(full_data_set, 5, train_det_model, "det_baseline")

    os.makedirs(f"saved_models/det_baseline_full", exist_ok=True)
    train_det_model(full_data_set, 300, f"saved_models/det_baseline_full")


def k_fold_validaton_det(dataset, k, train_func, experiment_name: str):
    kf = StratifiedKFold(n_splits=k)

    for ki, (train_idx, test_idx) in enumerate(kf.split(dataset, dataset.s_label[:, 0])):
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        results_folder = f"saved_models/{experiment_name}_k{ki}"
        os.makedirs(results_folder, exist_ok=True)
        kth_model = train_func(train_set, 300, results_folder)
        eval_metrics = eval_model_det(kth_model, test_set)
        print(
            f"K-{ki}\t BCE: {eval_metrics['bce']} Accuracy: {eval_metrics['accuracy']}, AUC-ROC: {eval_metrics['auc']}, Avg-Precision: {eval_metrics['avg_p']}")


def eval_model_det(model, test_set):
    combined_preds = torch.tensor([]).cuda()
    combined_trus = torch.tensor([]).cuda()

    test_loader = DataLoader(test_set, 8000)

    model.eval()
    for test_x, test_y in test_loader:
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        pred_test_y = model(test_x).view(-1)

        combined_preds = torch.cat((combined_preds, pred_test_y))
        combined_trus = torch.cat((combined_trus, test_y[:, 0]))

    combined_trus = combined_trus.detach().cpu().numpy()
    combined_preds = torch.sigmoid(combined_preds).detach().cpu().numpy()

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


def train_det_model(train_set: Dataset, epochs: int, results_folder: str):
    print(results_folder)
    print(f"Training Set Size: {len(train_set)}")

    b_size = 1024
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)

    model = PEMClass_Deterministic(14, 1, 20, use_cuda=True)
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training Step
    for j in range(epochs):
        model.train()
        train_losses = []
        for train_x, train_y in train_loader:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            optimizer.zero_grad()
            pred_y = model(train_x)
            loss = loss_fn(pred_y, train_y[:, [0]])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        if j % 10 == 0:
            print(f"[t:  {j + 1}] BCE: {train_loss:.4f}")

    save_model_det(model, join(results_folder, "pem_class_train_full"))
    return model


if __name__ == "__main__":
    run()
