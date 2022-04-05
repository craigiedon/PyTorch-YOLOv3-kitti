import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from utils.utils import load_classes


def discretize_truncation(trunc_val, buckets):
    return int(trunc_val * (buckets - 1))


def discretize_trunc_tensor(trunc_tensor, buckets):
    return (trunc_tensor * (buckets - 1)).to(dtype=torch.long)


def filter_inp_labels(inps, labels, filter_fn):
    zipped_filter = [(inp, l) for inp, l in zip(inps, labels) if filter_fn(inp, l)]
    filtered_inp, filtered_label = list(zip(*zipped_filter))
    return torch.tensor(np.array(filtered_inp)), torch.tensor(np.array(filtered_label))


def group_data(inps, labels, group_fn):
    inp_label_pairs = [(inp, lab) for inp, lab in zip(inps, labels)]
    groups = [(k, list(g)) for k, g in groupby(sorted(inp_label_pairs, key=group_fn), key=group_fn)]
    return groups


def count_tru_pos_v_false_neg(grouped_inp_labels):
    group_tps = []
    group_fns = []
    for k, g in grouped_inp_labels:
        g_inps, g_labs = list(zip(*g))

        fns = len([x for x in g_labs if x[0] < 0.1])
        tps = len(g_labs) - fns
        group_tps.append(tps)
        group_fns.append(fns)
    return np.array(group_tps), np.array(group_fns)


# class SalientObstacleDataset(Dataset):
#     def __init__(self, s_input_path, s_label_path):
#         # Format: <Class Num> <Truncation> <Occlusion> <alpha> <dim_w> <dim_l> <dim_h> <loc_x> <loc_y> <loc_z> <rot_y>
#         self.s_inp = torch.tensor(np.loadtxt(s_input_path))
#
#         self.s_label = torch.tensor(np.loadtxt(s_label_path))
#
#     def __len__(self):
#         return len(self.s_inp)
#
#     def __getitem__(self, index) -> T_co:
#         return self.s_inp[index], self.s_label[index]


def run():
    salient_input_path = "salient_dataset/salient_inputs_no_miscUnknown.txt"
    salient_label_path = "salient_dataset/salient_labels_no_miscUnknown.txt"

    s_inp = np.loadtxt(salient_input_path)
    s_label = np.loadtxt(salient_label_path)

    # Correlation between camera view x position and image view x position
    det_inp, det_label = filter_inp_labels(s_inp, s_label, lambda i, l: l[0] == 1)
    print("Total objects: ", len(s_inp))
    print("Total false negatives: ", len(s_inp) - len(det_inp))
    plt.scatter(det_inp[:, 7], det_label[:, 1])
    plt.xlabel("Camera X")
    plt.ylabel("Detection X")
    plt.show()

    # Correlation between camera view y position and image view y position
    plt.scatter(det_inp[:, 8], det_label[:, 2])
    plt.xlabel("Camera Y")
    plt.ylabel("Detection Y")
    plt.show()

    # Correlation between class name and detection rate
    class_groups = group_data(s_inp, s_label, lambda x: x[0][0])
    class_tps, class_fns = count_tru_pos_v_false_neg(class_groups)
    classes = load_classes("data/kitti.names")

    plt.bar(range(len(class_groups)), class_fns, label='False Neg')
    plt.bar(range(len(class_groups)), class_tps, bottom=class_fns, label='True Pos')
    plt.xlabel("Class Name")
    plt.xticks(range(len(class_groups)), labels=classes[:-1])
    plt.ylabel("Num Objects")
    plt.legend()
    plt.show()

    # Correlation between occlusion category and detection rate
    occlusion_groups = group_data(s_inp, s_label, lambda x: x[0][2])
    occlusion_tps, occlusion_fns = count_tru_pos_v_false_neg(occlusion_groups)

    plt.bar(range(len(occlusion_groups)), occlusion_fns, label='False Neg')
    plt.bar(range(len(occlusion_groups)), occlusion_tps, bottom=occlusion_fns, label='Tru Pos')
    plt.xlabel("Occlusion Type")
    plt.xticks(range(len(occlusion_groups)), labels=['Full Vis', 'Part Vis', 'Largely Occluded', 'Unknown'][:-1])
    plt.ylabel("Num Objects")
    plt.legend()
    plt.show()

    # Correlation between trunctation float and detection rate
    buckets = 4
    trunc_groups = group_data(s_inp, s_label, lambda x: discretize_truncation(x[0][1], buckets))
    trunc_tps, trunc_fns = count_tru_pos_v_false_neg(trunc_groups)

    plt.bar(range(len(trunc_groups)), trunc_fns, label="False Negs")
    plt.bar(range(len(trunc_groups)), trunc_tps, bottom=trunc_fns, label="Tru Pos")
    plt.xticks(range(len(trunc_groups)),
               labels=[f'{100.0 * n / buckets}-{100.0 * (n + 1) / buckets}%' for n in range(len(trunc_groups))])
    plt.xlabel("Truncation Amount")
    plt.ylabel("Num Objects")
    plt.legend()
    plt.show()

    # Q: What are the *outputs* of the NN in the "towards" paper?

    assert (len(s_inp) == len(s_label))

    # TODO: Create torch dataset


if __name__ == "__main__":
    run()
