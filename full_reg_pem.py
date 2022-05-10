import os
from os.path import join

import pyro
import torch
import numpy as np
from matplotlib import pyplot as plt
from pyro.infer import Trace_ELBO, SVI, Predictive, JitTraceMeanField_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoDiagonalNormal
from sklearn import linear_model, svm
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, Dataset

from load_salient_dataset import filter_inp_labels
from pems import SalientObstacleDataset, PEMReg_Deterministic, PEMReg_Aleatoric, PEMReg, save_model_bnn, save_model_det
import torch.nn.functional as F


def mean_square_dist_loss(input, target):
    assert input.dim() == target.dim() == 2
    assert input.shape == target.shape
    return torch.square(input - target).sum(1).mean(0)


def train_model(train_set: Dataset, epochs: int, results_folder: str):
    model = PEMReg(14, 2, use_cuda=True)
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model).cuda()

    adam = pyro.optim.Adam({"lr": 1e-3})
    loss_elb = Trace_ELBO(vectorize_particles=True)
    svi = SVI(model, guide, adam, loss=loss_elb)

    # Training Step
    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)

    for j in range(epochs):
        train_loss = 0
        for tx, ty in train_loader:
            tx = tx.cuda()
            ty = ty.cuda()
            err_label = ty[:, 5:7]

            tl = svi.step(tx, err_label)
            train_loss += tl

        train_loss = train_loss / len(train_set)

        if j % 10 == 0:
            print(f"[t:  {j + 1}] ELBO: {train_loss:.4f}")

    save_model_bnn(guide, pyro.get_param_store(), join(results_folder, "pem_reg_train_full"))


def eval_model(model, guide, test_set: Dataset):
    test_predictive = Predictive(model, guide=guide, num_samples=800, return_sites=("obs", "_RETURN"))
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
    avg_nlls = []
    for test_x, test_y in test_loader:
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        err_label = test_y[:, 5:7]
        preds = test_predictive(test_x)["_RETURN"]
        pred_mus = preds[:, 0].mean(0)
        pred_vars = torch.exp(2 * preds[:, 1]).mean(0)
        avg_nlls.append(F.gaussian_nll_loss(pred_mus, err_label, pred_vars).item())
    avg_test_nll = np.mean(avg_nlls)

    return avg_test_nll


def eval_model_al(model, test_set: Dataset):
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
    model.eval()
    val_losses = []
    mse_val_losses = []

    for tx, ty in test_loader:
        tx = tx.cuda()
        ty = ty.cuda()
        err_label = ty[:, 5:7]
        # pred_label = model(tx)
        pred_mus, pred_log_stds = model(tx)
        val_losses.append(F.gaussian_nll_loss(pred_mus, err_label, torch.exp(2.0 * pred_log_stds)).item())
        mse_val_losses.append(mean_square_dist_loss(pred_mus, err_label).item())
    mse_val = np.mean(mse_val_losses)
    avg_val_loss = np.mean(val_losses)

    return avg_val_loss


def train_model_aleatoric(train_set: Dataset, test_set: Dataset, epochs: int, results_folder: int):
    model = PEMReg_Aleatoric(14, 2, use_cuda=True)

    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)

    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)

    epoch_t_losses = []
    epoch_v_losses = []

    loss_fn = torch.nn.GaussianNLLLoss()
    optimizer = Adam(model.parameters())

    # Train step
    for i in range(epochs):
        model.train()
        train_losses = []
        mse_train_losses = []
        for tx, ty in train_loader:
            tx = tx.cuda()
            ty = ty.cuda()
            optimizer.zero_grad()
            err_label = ty[:, 5:7]
            pred_mus, pred_log_stds = model(tx)
            loss = loss_fn(pred_mus, err_label, torch.exp(2.0 * pred_log_stds))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            mse_train_losses.append(mean_square_dist_loss(pred_mus, err_label).item())
        mse_train = np.mean(mse_train_losses)
        avg_train_loss = np.mean(train_losses)

        if i % 10 == 0:
            # Validation Step
            if test_set is not None:
                model.eval()
                val_losses = []
                mse_val_losses = []
                for tx, ty in test_loader:
                    tx = tx.cuda()
                    ty = ty.cuda()
                    err_label = ty[:, 5:7]
                    # pred_label = model(tx)
                    pred_mus, pred_log_stds = model(tx)
                    loss = loss_fn(pred_mus, err_label, torch.exp(2.0 * pred_log_stds))
                    val_losses.append(loss.item())
                    mse_val_losses.append(mean_square_dist_loss(pred_mus, err_label).item())
                mse_val = np.mean(mse_val_losses)
                avg_val_loss = np.mean(val_losses)

                print(f"E-{i}, NLL: {avg_train_loss}, NLL-V: {avg_val_loss}, MSE-T: {mse_train}, MSE-V {mse_val}")
            else:
                print(f"E-{i}, NLL: {avg_train_loss}, MSE-T: {mse_train}")

            # epoch_t_losses.append(avg_train_loss)
            # epoch_v_losses.append(avg_val_loss)

    save_model_det(model, f"{results_folder}/pem_reg_al_full")

    # plt.plot(range(len(epoch_t_losses)), epoch_t_losses, label="Train")
    # plt.plot(range(len(epoch_v_losses)), epoch_v_losses, label="Val")
    # plt.xlabel("Epochs")
    # plt.ylabel("Gaussian NLL")
    # plt.legend()
    # plt.savefig(f"{results_folder}/training_loss.pdf")


def linear_baseline(train_set: Dataset, test_set: Dataset):
    training_ins = torch.stack([tr[0] for tr in train_set])
    training_labels = torch.stack([tr[1][5:7] for tr in train_set])

    test_ins = torch.stack([ts[0] for ts in test_set])
    test_labels = torch.stack([ts[1][5:7] for ts in test_set])

    data_vars = torch.var(training_labels, axis=0).repeat(len(test_labels), 1)

    linear_model = Ridge()
    linear_model.fit(training_ins, training_labels)
    linear_preds = linear_model.predict(test_ins)
    # linear_mse = mean_square_dist_loss(torch.tensor(linear_preds), test_labels)

    linear_nll = F.gaussian_nll_loss(torch.from_numpy(linear_preds), test_labels, data_vars)
    return linear_nll


def guess_mu_baseline(train_set: Dataset, test_set: Dataset):
    training_ins = torch.stack([tr[0] for tr in train_set])
    training_labels = torch.stack([tr[1][5:7] for tr in train_set])
    test_ins = torch.stack([ts[0] for ts in test_set])
    test_labels = torch.stack([ts[1][5:7] for ts in test_set])

    data_mus = torch.mean(training_labels, axis=0).repeat(len(test_labels), 1)
    data_vars = torch.var(training_labels, axis=0).repeat(len(test_labels), 1)

    summary_mse = mean_square_dist_loss(data_mus, test_labels)
    summary_nll = F.gaussian_nll_loss(data_mus, test_labels, data_vars)
    return summary_nll


def run():
    s_inputs = np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt")
    s_labels = np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt")
    s_inputs, s_labels = filter_inp_labels(s_inputs, s_labels, lambda i, l: l[0] == 1)
    s_inputs, s_labels = torch.tensor(s_inputs, dtype=torch.float), torch.tensor(s_labels, dtype=torch.float)

    skip = 1
    s_inputs, s_labels = s_inputs[::skip], s_labels[::skip]

    full_data_set = SalientObstacleDataset(s_inputs, s_labels)

    kf = KFold(n_splits=5)
    epochs = 300

    for ki, (train_idx, test_idx) in enumerate(kf.split(full_data_set)):
        print(f"Fold: {ki}")
        train_set = Subset(full_data_set, train_idx)
        test_set = Subset(full_data_set, test_idx)

        # svm_baseline(train_set, test_set)

        # print("BNN Training...")
        # results_folder_bnn = f"saved_models/bnn_reg_k{ki}"
        # os.makedirs(results_folder_bnn, exist_ok=True)
        # train_model(train_set, epochs, results_folder_bnn)

        print("Aleatoric Training...")
        results_folder_al = f"saved_models/al_reg_k{ki}"
        os.makedirs(results_folder_al, exist_ok=True)
        train_model_aleatoric(train_set, test_set, epochs, results_folder_al)

    # print("Full BNN Training...")
    # os.makedirs(f"saved_models/bnn_reg_full", exist_ok=True)
    # train_model(full_data_set, epochs, f"saved_models/bnn_reg_full")

    print("Full Aleatoric Training...")
    os.makedirs(f"saved_models/al_reg_full", exist_ok=True)
    train_model_aleatoric(full_data_set, None, epochs, f"saved_models/al_reg_full")


if __name__ == "__main__":
    run()
