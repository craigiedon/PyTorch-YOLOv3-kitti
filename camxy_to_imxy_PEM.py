# Load in the data
# F.onehot will give you the encodings you desire

### Vehicles to false negatives
### Start with the standard torch NNs, then upgrade to a pyro one?

import numpy as np
import pyro
import torch
from pyro.nn import PyroModule
import pyro.distributions as dist
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits import mplot3d

from load_salient_dataset import filter_inp_labels
from pems import PEMReg, SalientObstacleDataset


def run():
    model = PEMReg(2, 2, 20)
    model.cuda()

    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"
    s_inputs = np.loadtxt(salient_input_path)
    s_labels = np.loadtxt(salient_label_path)
    s_inputs, s_labels = filter_inp_labels(s_inputs, s_labels, lambda i, l: l[0] == 1)

    s_inputs = torch.tensor(s_inputs, dtype=torch.float)[::10, [7, 8]]
    s_labels = torch.tensor(s_labels, dtype=torch.float)[::10, [1, 2]]

    in_means = torch.mean(s_inputs, 0)
    in_stds = torch.std(s_inputs, 0)

    print(in_means.shape)

    s_inputs = (s_inputs - in_means) / in_stds

    s_labels[:, 0] = s_labels[:, 0] / 1000
    s_labels[:, 1] = s_labels[:, 1] / 1000

    data_loader = DataLoader(SalientObstacleDataset(s_inputs, s_labels), batch_size=1024, shuffle=True)

    # x_data = torch.linspace(-1, 1, 40)
    # y_data = torch.cat((torch.zeros(20), torch.ones(20))).to(torch.long)

    # Define loss and optimize
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.GaussianNLLLoss()
    optim = torch.optim.Adam(model.parameters())
    num_iterations = 100

    for j in range(num_iterations):
        for salient_x, salient_y in data_loader:
            salient_x = salient_x.cuda()
            salient_y = salient_y.cuda()

            y_pred = model(salient_x)
            y_tru = salient_y

            loss = loss_fn(y_pred[0], salient_y, torch.exp(2 * y_pred[1]))
            optim.zero_grad()
            loss.backward()
            optim.step()
        if j % 10 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

            # run the model forward on the data
            # y_pred = model(x_data)
            # # calculate the mse loss
            # loss = loss_fn(y_pred, y_data)
            # # initialize gradients to zero
            # optim.zero_grad()
            # # backpropagate
            # loss.backward()
            # # take a gradient step
            # optim.step()

    # Inspect learned parameters
    # print("Learned parameters:")
    # for name, param in model.named_parameters():
    #     print(name, param.data.numpy())

    model.eval()
    # xs = torch.linspace(-3.5, 3.5, 1000)
    # inp_valid = torch.cartesian_prod(xs, xs)

    y_pred_mu, y_pred_log_sig = model(s_inputs.cuda())
    y_pred_mu = y_pred_mu.detach().cpu().numpy()
    y_pred_sig = torch.exp(y_pred_log_sig).detach().cpu().numpy()

    fig = plt.figure()

    axs = [fig.add_subplot(2, 1, i, projection='3d') for i in range(1, 3)]
    skip = 10
    s_in_np = s_inputs.detach().numpy()[::skip]

    axs[0].scatter(s_in_np[:, 0], s_in_np[:, 1], s_labels.detach().numpy()[::skip, 0], marker='o', s=2)
    axs[0].scatter(s_in_np[:, 0], s_in_np[:, 1], y_pred_mu[::skip, 0], marker='x', s=2)

    axs[1].scatter(s_in_np[:, 0], s_in_np[:, 1], s_labels.detach().numpy()[::skip, 1], marker='o', s=2)
    axs[1].scatter(s_in_np[:, 0], s_in_np[:, 1], y_pred_mu[::skip, 1], marker='x', s=2)

    # plt.scatter(s_inputs.detach().numpy(), s_labels.detach().numpy()) plt.plot(x_valid.detach().cpu().numpy(),
    # y_pred_mu, color='orange') plt.fill_between(x_valid.detach().cpu().numpy().reshape(-1), (y_pred_mu -
    # y_pred_sig).reshape(-1), (y_pred_mu + y_pred_sig).reshape(-1), color='orange', alpha=0.3) plt.show()

    # yes_xs = [x for x, y in zip(x_data, y_data) if y == 1]
    # no_xs = [x for x, y in zip(x_data, y_data) if y == 0]
    #
    # yes_pred_xs = [x for x, y in zip(x_data, y_pred) if y[1] > y[0]]
    # no_pred_xs = [x for x, y in zip(x_data, y_pred) if y[0] > y[1]]
    #
    # plt.scatter(yes_xs, torch.ones(len(yes_xs)), color='b')
    # plt.scatter(no_xs, torch.ones(len(no_xs)), color='orange')
    #
    # plt.scatter(yes_pred_xs, torch.zeros(len(yes_xs)), color='b')
    # plt.scatter(no_pred_xs, torch.zeros(len(no_xs)), color='orange')

    plt.show()


if __name__ == "__main__":
    run()
