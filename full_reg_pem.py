import os
from os.path import join

import pyro
import torch
import numpy as np
from matplotlib import pyplot as plt
from pyro.infer import Trace_ELBO, SVI, Predictive, JitTraceMeanField_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoDiagonalNormal
from sklearn import linear_model, svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader, Subset

from load_salient_dataset import filter_inp_labels
from pems import SalientObstacleDataset, PEMReg_Deterministic, PEMReg_Aleatoric, PEMReg, save_model_bnn
import torch.nn.functional as F


def mean_square_dist_loss(input, target):
    assert input.dim() == target.dim() == 2
    assert input.shape == target.shape
    return torch.square(input - target).sum(1).mean(0)


def run():
    s_inputs = np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt")
    s_labels = np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt")
    s_inputs, s_labels = filter_inp_labels(s_inputs, s_labels, lambda i, l: l[0] == 1)
    s_inputs, s_labels = torch.tensor(s_inputs, dtype=torch.float), torch.tensor(s_labels, dtype=torch.float)

    skip = 1
    s_inputs, s_labels = s_inputs[::skip], s_labels[::skip]

    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    print(full_data_set)

    kf = KFold(n_splits=5)
    # loss_fn = mean_square_dist_loss

    for ki, (train_idx, test_idx) in enumerate(kf.split(full_data_set)):
        print(f"Fold: {ki}")
        train_set = Subset(full_data_set, train_idx)
        test_set = Subset(full_data_set, test_idx)
        results_folder = f"saved_models/bnn_reg_k{ki}"
        os.makedirs(results_folder, exist_ok=True)

        print(f"Train Size: {len(train_set)}, Test Size: {len(test_set)}")

        training_ins = full_data_set.final_ins[train_idx]
        training_labels = full_data_set.s_label[train_idx, 5:7]  # [5, 6]]

        test_ins = full_data_set.final_ins[test_idx]
        test_labels = full_data_set.s_label[test_idx, 5:7]
        error_sizes = np.linalg.norm(test_labels, axis=1)
        max_eid = np.argmax(error_sizes)

        print(
            f"Error Mean: {np.mean(error_sizes)} Error Std: {np.std(error_sizes)} Max Error: {test_labels[max_eid]}, mag: {error_sizes[max_eid]}")
        # print(f"Sanity check: {torch.sqrt(loss_fn(torch.zeros(1, 2), test_labels[max_eid].view(1, 2)))}")

        reg = linear_model.Ridge()
        reg.fit(training_ins, training_labels)
        reg_preds = reg.predict(test_ins)
        lin_mse = mean_square_dist_loss(torch.tensor(reg_preds), test_labels)
        data_mus = torch.mean(training_labels, axis=0).repeat(len(test_labels), 1)
        data_vars = torch.var(training_labels, axis=0).repeat(len(test_labels), 1)
        lin_nll = F.gaussian_nll_loss(torch.tensor(reg_preds), test_labels, data_vars)
        print(
            f"Linear Reg MSE: {lin_mse}, NLL: {lin_nll}, Avg Pred Dist: {np.mean(np.linalg.norm(reg_preds, axis=1))} Std: {np.std(np.linalg.norm(reg_preds, axis=1))}")

        summary_mse = mean_square_dist_loss(data_mus, test_labels)
        summary_nll = F.gaussian_nll_loss(data_mus, test_labels, data_vars)
        print(f"Summary MSE: {summary_mse}, NLL: {summary_nll}")

        zero_mse = mean_square_dist_loss(torch.zeros(len(test_labels), 2), test_labels)
        zero_nll = F.gaussian_nll_loss(torch.zeros(test_labels.shape), test_labels, torch.ones(test_labels.shape))
        print(f"Guess-Zero MSE: {zero_mse} NLL: {zero_nll}")
        #
        # loc = torch.mean(test_labels, dim=0)
        # scl = torch.std(test_labels, dim=0)
        # rand_dist = torch.distributions.Normal(loc=loc, scale=scl)
        # rand_preds = rand_dist.sample_n(len(test_labels))
        # rand_mse = loss_fn(torch.tensor(rand_preds), test_labels)
        # print(f"Rand MSE: {rand_mse}")
        #
        # svm_model = MultiOutputRegressor(svm.SVR())
        # svm_model.fit(training_ins, training_labels)
        # svm_preds = svm_model.predict(test_ins)
        # svm_mse = loss_fn(torch.tensor(svm_preds), test_labels)
        # print(f"SVM MSE: {svm_mse}")

        # model = PEMReg_Deterministic(14, 2, h=10, use_cuda=True)
        # model = torch.nn.Linear(14, 2).cuda()
        # model = PEMReg_Aleatoric(14, 2, use_cuda=True)
        model = PEMReg(14, 2, use_cuda=True)
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(model).cuda()

        adam = pyro.optim.Adam({"lr": 1e-3})
        loss_elb = JitTraceMeanField_ELBO(vectorize_particles=True)
        svi = SVI(model, guide, adam, loss=loss_elb)

        # Training Step
        train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
        epochs = 5000

        for j in range(epochs):
            train_loss = 0
            for tx, ty in train_loader:
                tx = tx.cuda()
                ty = ty.cuda()
                err_label = ty[:, 5:7]

                tl = svi.step(tx, err_label)
                # print(tl)
                train_loss += tl

            train_loss = train_loss / len(train_set)

            if j % 10 == 0:
                print(f"[t:  {j + 1}] ELBO: {train_loss:.4f}")
            if j % 100 == 0:
                test_predictive = Predictive(model, guide=guide, num_samples=800, return_sites=("obs", "_RETURN"))
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
                print(f"NLL: {avg_test_nll}")


        save_model_bnn(guide, pyro.get_param_store(), join(results_folder, "pem_reg_train_full"))

        # optimizer = torch.optim.Adam(model.parameters())
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)

        # epoch_t_losses = []
        # epoch_v_losses = []
        #
        # loss_fn = torch.nn.GaussianNLLLoss()
        #
        # for i in range(epochs):
        #     model.train()
        #     train_losses = []
        #     mse_train_losses = []
        #     for tx, ty in train_loader:
        #         tx = tx.cuda()
        #         ty = ty.cuda()
        #         optimizer.zero_grad()
        #         err_label = ty[:, 5:7]
        #         pred_mus, pred_log_stds = model(tx)
        #         loss = loss_fn(pred_mus, err_label, torch.exp(2.0 * pred_log_stds))
        #         loss.backward()
        #         optimizer.step()
        #         train_losses.append(loss.item())
        #         mse_train_losses.append(mean_square_dist_loss(pred_mus, err_label).item())
        #     mse_train = np.mean(mse_train_losses)
        #     avg_train_loss = np.mean(train_losses)
        #
        #     if i % 10 == 0:
        #         model.eval()
        #         val_losses = []
        #         mse_val_losses = []
        #         for tx, ty in test_loader:
        #             tx = tx.cuda()
        #             ty = ty.cuda()
        #             optimizer.zero_grad()
        #             err_label = ty[:, 5:7]
        #             # pred_label = model(tx)
        #             pred_mus, pred_log_stds = model(tx)
        #             loss = loss_fn(pred_mus, err_label, torch.exp(2.0 * pred_log_stds))
        #             loss.backward()
        #             optimizer.step()
        #             val_losses.append(loss.item())
        #             mse_val_losses.append(mean_square_dist_loss(pred_mus, err_label).item())
        #         mse_val = np.mean(mse_val_losses)
        #         avg_val_loss = np.mean(val_losses)
        #
        #         print(
        #             f"E-{i}, NLL: {avg_train_loss}, NLL-V: {avg_val_loss}, MSE-T: {mse_train}, MSE-V {mse_val}")  # , lr: {scheduler.get_last_lr()}")
        #
        #         epoch_t_losses.append(avg_train_loss)
        #         epoch_v_losses.append(avg_val_loss)
        #         # print(f"lr: {scheduler.get_last_lr()}")
        #
        #     # if i < 1001:
        #     # scheduler.step()
        #
        # # epoch_v_losses = np.zeros(100)
        #
        # plt.plot(range(len(epoch_t_losses)), epoch_t_losses, label="Train")
        # plt.plot(range(len(epoch_v_losses)), epoch_v_losses, label="Val")
        # plt.plot(range(len(epoch_v_losses)), np.repeat(np.sqrt(lin_mse), len(epoch_v_losses)), linestyle='dashed', label="Lin Reg")
        # plt.plot(range(len(epoch_v_losses)), np.repeat(np.sqrt(zero_mse), len(epoch_v_losses)), linestyle='dashed', label="Zero-Guess")
        # plt.plot(range(len(epoch_v_losses)), np.repeat(np.sqrt(svm_mse), len(epoch_v_losses)), linestyle='dashed', label="SVM (RBF)")
        # plt.plot(range(len(epoch_v_losses)), np.repeat(np.sqrt(rand_mse), len(epoch_v_losses)), linestyle='dashed', label="Rand Fuzz")

        plt.xlabel("Epochs")
        plt.ylabel("Gaussian NLL")
        # plt.ylabel("RMSE")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    run()
