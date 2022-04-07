import numpy as np

from load_salient_dataset import filter_inp_labels

salient_input_path = "salient_dataset/salient_inputs.txt"
salient_label_path = "salient_dataset/salient_labels.txt"

s_inp = np.loadtxt(salient_input_path)
s_label = np.loadtxt(salient_label_path)

# Filter out Class 7 ("Miscellaneous") and Occlusion level 2 ("Unknown")
filtered_inp, filtered_label = filter_inp_labels(s_inp, s_label, filter_fn=lambda i, l: i[0] != 7 and i[2] != 3)

np.savetxt("salient_dataset/salient_inputs_no_miscUnknown.txt", filtered_inp, fmt=['%.0f', "%.3f", "%.0f", "%.3f","%.3f", "%.3f", "%.3f", "%.3f", "%.3f", "%.3f", "%.3f"],
           header="Format: <Class Num> <Truncation> <Occlusion> <alpha> <dim_w> <dim_l> <dim_h> <loc_x> <loc_y> <loc_z> <rot_y>")
np.savetxt("salient_dataset/salient_labels_no_miscUnknown.txt", filtered_label, fmt='%.0f',
           header="Format: <Detected> <bbox cx> <bbox cy> <bbox_w> <bbox_h> <err cx> <err cy> <err w> <err h>")