from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from load_salient_dataset import group_data, count_tru_pos_v_false_neg, discretize_truncation, discretize_trunc_tensor, \
    filter_inp_labels
from pems import PEMClass, SalientObstacleDataset

import numpy as np
import torch
import torch.nn.functional as F


def run():
    model = PEMClass(1, 1, 20)
    model.cuda()

    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"
    s_inputs = np.loadtxt(salient_input_path)
    s_labels = np.loadtxt(salient_label_path)

    s_inputs, s_labels = filter_inp_labels(s_inputs, s_labels, lambda i, l: i[0] == 3)
    print(len(s_inputs))

    s_inputs = torch.tensor(s_inputs, dtype=torch.float)[::1, [1]]  # Class label
    s_labels = torch.tensor(s_labels, dtype=torch.float)[::1, [0]]  # Detected Y/N

    class_grouped = group_data(s_inputs, s_labels, lambda x: discretize_truncation(x[0][0], 10))
    group_tps, group_fns = count_tru_pos_v_false_neg(class_grouped)
    class_tp_props = group_tps / (group_tps + group_fns)
    plt.plot(range(len(class_tp_props)), class_tp_props)
    plt.show()

    data_loader = DataLoader(SalientObstacleDataset(s_inputs, s_labels), batch_size=1024, shuffle=True)

    # Define loss and optimize
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters())
    num_iterations = 1000

    for j in range(num_iterations):
        for salient_x, salient_y in data_loader:
            salient_x = salient_x.cuda()
            salient_y = salient_y.cuda()

            y_pred = model(salient_x)
            y_tru = salient_y

            loss = loss_fn(y_pred.view(-1), y_tru.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
        if j % 10 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

    model.eval()

    xs_val = torch.linspace(0, 1, 100)
    y_pred = model(xs_val)
    y_pred = y_pred.detach().cpu().numpy()

    plt.plot(xs_val, y_pred)
    plt.show()


if __name__ == "__main__":
    run()
