import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby


def discretize_trunctation(trunc_val):
    return int(trunc_val * 3)


salient_input_path = "salient_dataset/salient_inputs.txt"
salient_label_path = "salient_dataset/salient_labels.txt"

s_inp = np.loadtxt(salient_input_path)
s_label = np.loadtxt(salient_label_path)

# detected_inp_label = [(inp, lab) for inp, lab in zip(s_inp, s_label) if lab[0] == 1]
detected_inp_label = [(inp, lab) for inp, lab in zip(s_inp, s_label)]
det_inp, det_label = list(zip(*detected_inp_label))
det_inp = np.array(det_inp)
det_label = np.array(det_label)

# Correlation between camera view x position and image view x position
plt.scatter(det_inp[:, 7], det_label[:, 1])
plt.xlabel("Camera X")
plt.ylabel("Detection X")
plt.show()
#

# Correlation between camera view y position and image view y position
plt.scatter(det_inp[:, 8], det_label[:, 2])
plt.xlabel("Camera Y")
plt.ylabel("Detection Y")
plt.show()

# Correlation between class name and detection rate
class_groups = [(k, list(g)) for k, g in groupby(sorted(detected_inp_label, key=lambda x: x[0][0]), key=lambda x: x[0][0])]
class_props = []
for k, g in class_groups:
    print("Group: ", k)
    g_inps, g_labs = list(zip(*g))

    false_negs = [x for x in g_labs if x[0] < 0.1]
    class_prop = 100.0 * len(false_negs) / len(g_labs)
    class_props.append(class_prop)

plt.bar(range(len(class_groups)), class_props)
plt.xlabel("Class Name")
plt.ylabel("False Negative Percentage")
plt.show()

# Correlation between occlusion category and detection rate
occlusion_groups = [(k, list(g)) for k, g in groupby(sorted(detected_inp_label, key=lambda x: x[0][2]), key=lambda x: x[0][2])]
print(type(occlusion_groups))
fn_props = []
for k, g in occlusion_groups:
    print("Group: ", k)
    g_inps, g_labs = list(zip(*g))

    false_negs = [x for x in g_labs if x[0] < 0.1]
    fn_prop = 100.0 * len(false_negs) / len(g_labs)
    fn_props.append(fn_prop)
    print(f"Size: {len(g_labs)}, FNs: {len(false_negs)} FN-Prop: {fn_prop}")

plt.bar([0, 1, 2, 3], fn_props)
plt.xlabel("Occlusion Type")
plt.ylabel("False Negative Percentage")
plt.show()

# Correlation between trunctation float and detection rate
truncation_groups = [(k, list(g)) for k, g in groupby(sorted(detected_inp_label, key=lambda x: discretize_trunctation(x[0][1])), key=lambda x: discretize_trunctation(x[0][1]))]
trunc_fns = []
for k, g in truncation_groups:
    print("Group: ", k)
    g_inps, g_labs = list(zip(*g))

    false_negs = [x for x in g_labs if x[0] < 0.1]
    fn_prop = 100.0 * len(false_negs) / len(g_labs)
    trunc_fns.append(fn_prop)
    print(f"Size: {len(g_labs)}, FNs: {len(false_negs)} FN-Prop: {fn_prop}")

plt.bar(range(4), trunc_fns)
plt.xlabel("Truncation Amount")
plt.ylabel("False Negative Percentage")
plt.show()

# Q: What are the *outputs* of the NN in the "towards" paper?

assert (len(s_inp) == len(s_label))

# TODO: Create torch dataset
