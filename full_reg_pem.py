import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, Subset

from load_salient_dataset import filter_inp_labels
from pems import SalientObstacleDataset, PEMReg_Deterministic
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

    skip = 5
    s_inputs, s_labels = s_inputs[::skip], s_labels[::skip]

    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    print(full_data_set)

    kf = KFold(n_splits=5)
    loss_fn = mean_square_dist_loss

    for ki, (train_idx, test_idx) in enumerate(kf.split(full_data_set)):
        print(f"Fold: {ki}")
        train_set = Subset(full_data_set, train_idx)
        test_set = Subset(full_data_set, test_idx)

        print(f"Train Size: {len(train_set)}, Test Size: {len(test_set)}")

        training_ins = full_data_set.final_ins[train_idx]
        training_labels = full_data_set.s_label[train_idx, 5:7]  # [5, 6]]

        test_ins = full_data_set.final_ins[test_idx]
        test_labels = full_data_set.s_label[test_idx, 5:7]
        error_sizes = np.linalg.norm(test_labels, axis=1)
        max_eid = np.argmax(error_sizes)

        print(f"Max Error: {test_labels[max_eid]}, mag: {error_sizes[max_eid]}")
        print(f"Sanity check: {torch.sqrt(loss_fn(torch.zeros(1, 2), test_labels[max_eid].view(1, 2)))}")

        reg = linear_model.Ridge()

        reg.fit(training_ins, training_labels)
        reg_preds = reg.predict(test_ins)

        lin_mse = loss_fn(torch.tensor(reg_preds), test_labels)
        print(f"Linear Reg MSE: {lin_mse}")

        zero_mse = loss_fn(torch.zeros(len(test_labels), 2), test_labels)
        print(f"Guess-Zero MSE: {zero_mse}")

        model = PEMReg_Deterministic(14, 2, h=50, use_cuda=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)

        epochs = 10000
        epoch_t_losses = []
        epoch_v_losses = []

        train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)

        for i in range(epochs):
            model.train()
            train_losses = []
            for tx, ty in train_loader:
                tx = tx.cuda()
                ty = ty.cuda()
                optimizer.zero_grad()
                err_label = ty[:, 5:7]
                pred_label = model(tx)
                loss = loss_fn(pred_label, err_label)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)

            if i % 100 == 0:
                model.eval()
                val_losses = []
                for tx, ty in test_loader:
                    tx = tx.cuda()
                    ty = ty.cuda()
                    optimizer.zero_grad()
                    err_label = ty[:, 5:7]
                    pred_label = model(tx)
                    loss = loss_fn(pred_label, err_label)
                    loss.backward()
                    optimizer.step()
                    val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses)

                print(f"E-{i}, MSE-T: {avg_train_loss}, MSE-V: {avg_val_loss}, lr: {scheduler.get_last_lr()}")

                epoch_t_losses.append(avg_train_loss)
                epoch_v_losses.append(avg_val_loss)
                # print(f"lr: {scheduler.get_last_lr()}")

            if i < 1001:
                scheduler.step()

        plt.plot(range(len(epoch_t_losses)), epoch_t_losses, label="Train")
        plt.plot(range(len(epoch_v_losses)), epoch_v_losses, label="Val")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    run()
