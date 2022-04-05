from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from load_salient_dataset import group_data, count_tru_pos_v_false_neg
from pems import PEMClass, SalientObstacleDataset

import numpy as np
import torch
import torch.nn.functional as F


def run():
    model = PEMClass(3, 1, 20)
    model.cuda()

    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"
    s_inputs = np.loadtxt(salient_input_path)
    s_labels = np.loadtxt(salient_label_path)

    s_inputs = torch.tensor(s_inputs, dtype=torch.float)[::1, [2]]  # Class label
    s_labels = torch.tensor(s_labels, dtype=torch.float)[::1, [0]]  # Detected Y/N

    class_grouped = group_data(s_inputs, s_labels, lambda x: x[0][0])
    group_tps, group_fns = count_tru_pos_v_false_neg(class_grouped)
    class_tp_props = group_tps / (group_tps + group_fns)
    print(class_tp_props)

    s_inputs = F.one_hot(s_inputs.to(dtype=torch.long), 3).to(dtype=torch.float)

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

    one_hot_classes = F.one_hot(torch.arange(0, 3), 3).to(dtype=torch.float).cuda()
    y_pred = torch.sigmoid(model(one_hot_classes))
    y_pred = y_pred.detach().cpu().numpy()

    for cls, yp in enumerate(y_pred):
        print(f'Class: {cls}, Pred TP%: {yp.item():.3f}, Tru TP%: {class_tp_props[cls]:.3f}')

    plt.show()


if __name__ == "__main__":
    run()
