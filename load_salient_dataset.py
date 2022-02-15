import numpy as np
import matplotlib.pyplot as plt

salient_input_path = "salient_dataset/salient_inputs.txt"
salient_label_path = "salient_dataset/salient_labels.txt"

s_inp = np.loadtxt(salient_input_path)
s_label = np.loadtxt(salient_label_path)

detected_inp_label = [(inp, lab) for inp, lab in zip(s_inp, s_label) if lab[0] == 1]
det_inp, det_label = list(zip(*detected_inp_label))
det_inp = np.array(det_inp)
det_label = np.array(det_label)

# Correlation between camera view x position and image view x position
# plt.scatter(det_inp[:, 7], det_label[:, 1])
# plt.show()
#

# Correlation between camera view y position and image view y position
# plt.scatter(det_inp[:, 8], det_label[:, 2])
# plt.show()

# Correlation between class name and detection rate
plt.scatter(s_inp[:, 0], s_label[:, 0])
plt.show()

# Correlation between occlusion category and detection rate

# Correlation between trunctation float and detection rate

# Q: What are the *outputs* of the NN in the "towards" paper?

assert(len(s_inp) == len(s_label))

# TODO: Create torch dataset