import pyro
from matplotlib import pyplot as plt
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI, Predictive
import pyro
import pyro.distributions as dist
from torch.utils.data import DataLoader

from load_salient_dataset import group_data, count_tru_pos_v_false_neg, discretize_truncation, discretize_trunc_tensor, \
    filter_inp_labels
from pems import PEMClass, SalientObstacleDataset
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal

import numpy as np
import torch
import torch.nn.functional as F


def run():
    model = PEMClass(1, 2, 20, use_cuda=True)

    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"
    s_inputs = np.loadtxt(salient_input_path)
    s_labels = np.loadtxt(salient_label_path)

    s_inputs = torch.tensor(s_inputs, dtype=torch.float)[::50, [10]]  # Class label
    print(s_inputs.mean())
    print(s_inputs.std())
    s_inputs = (s_inputs - s_inputs.mean()) / s_inputs.std()
    s_labels = torch.tensor(s_labels, dtype=torch.float)[::50, [0]]  # Detected Y/N

    data_loader = DataLoader(SalientObstacleDataset(s_inputs, s_labels), batch_size=1024, shuffle=True)

    pyro.clear_param_store()
    # pyro.render_model(model, model_args=(s_inputs[:200].cuda(), s_labels[:200].cuda())).view()

    num_iterations = 5000
    guide = AutoMultivariateNormal(poutine.block(model, hide=['obs'])).cuda()
    p_store = pyro.get_param_store()
    adam = pyro.optim.Adam({"lr": 1e-3})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(vectorize_particles=True))

    for j in range(num_iterations):
        loss = 0
        for salient_x, salient_y in data_loader:
            salient_x = salient_x.cuda()
            salient_y = salient_y.cuda()

            loss += svi.step(salient_x, salient_y.view(-1))
        loss = loss / len(s_inputs)
        if j % 10 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss))

    # Define loss and optimize
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # optim = torch.optim.Adam(model.parameters())
    #
    # for j in range(num_iterations):
    #     for salient_x, salient_y in data_loader:
    #         salient_x = salient_x.cuda()
    #         salient_y = salient_y.cuda()
    #
    #         y_pred = model(salient_x)
    #         y_tru = salient_y
    #
    #         loss = loss_fn(y_pred.view(-1), y_tru.view(-1))
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #     if j % 10 == 0:
    #         print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

    # net.eval()

    predictive = Predictive(model, guide=guide, num_samples=2000, return_sites={"obs", "_RETURN"})
    x_pred_points = torch.linspace(-1, 1, 100).view(-1, 1).cuda()
    preds = predictive(x_pred_points)

    pred_probs_means = preds["obs"].to(dtype=torch.float).mean(0)
    pred_probs_std = preds["obs"].to(dtype=torch.float).std(0)

    # pred_probs = F.softmax(preds["_RETURN"], dim=2)
    # pred_probs_means = pred_probs.mean(0)[:, 1]
    # pred_probs_std = pred_probs.std(0)[:, 1]
    # print(pred_probs.shape)

    xpp = x_pred_points.view(-1).detach().cpu().numpy()
    po_mu = pred_probs_means.detach().cpu().numpy().reshape(-1)
    po_std = pred_probs_std.detach().cpu().numpy().reshape(-1)

    plt.plot(xpp, po_mu)
    plt.fill_between(xpp, po_mu - po_std, po_mu + po_std, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    run()
