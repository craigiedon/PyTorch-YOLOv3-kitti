from calib_project_vis import parse_kitti_label
from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

kitti_weights = 'weights/yolov3-kitti.weights'
labels_folder = "/home/craig/data/KITTI/labels/training"
model_config_path = "config/yolov3-kitti.cfg"
img_folder = '/home/craig/data/KITTI/images/training'
img_size = 416
os.makedirs('salient_dataset', exist_ok=True)

model = Darknet(model_config_path, img_size=img_size)
model.load_weights(kitti_weights)
model.cuda()
model.eval()

dataloader = DataLoader(ImageFolder(img_folder, img_size=img_size),
                        batch_size=4, shuffle=False, num_workers=16)

classes = load_classes('data/kitti.names')

img_filenames = sorted(os.listdir(img_folder))
img_detections = torch.load("salient_dataset/detections.pt")

label_paths = [os.path.join(labels_folder, f_name) for f_name in sorted(os.listdir(labels_folder))]

tru_detections = [parse_kitti_label(lp) for lp in label_paths]

# Start in the label folder
# For each file in this folder
# Parse out the infos, load them into a big list
# In the "running through YOLO" part, lets compare the number of detections from YOLO to the number of true detections
# Do some sort of hungarian algorithm to match them up (write down how you will deal with false negatives / positives"
# Have two files: salient_variables.txt and predictions.txt. They should be exactly the same number of lines! (And it should be *more* lines than just the number of images)
salient_variable_dataset = None

# for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
#     input_imgs = Variable(input_imgs.type(torch.cuda.FloatTensor))
#
#     with torch.no_grad():
#         detections = model(input_imgs)
#         detections = non_max_suppression(detections, 80, 0.8, 0.4)
#
#         print(f'Batch {batch_i}')
#
#         # imgs.extend(img_paths)
#         img_detections.extend(detections)

# Visualize detections
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
for img_filename, img_det, tru_det in zip(img_filenames, img_detections, tru_detections):
    print(img_filename)
    img = np.array(Image.open(os.path.join(img_folder, img_filename)))
    fig, (ax_1, ax_2) = plt.subplots(2)
    ax_1.imshow(img)
    ax_2.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))

    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if img_det is None:
        raise ValueError

    print(img_det.size())
    unique_labels = img_det[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    print("Class predictions:", n_cls_preds)
    bbox_colors = random.sample(colors, n_cls_preds)

    for d in tru_det:
        x_min, y_min, x_max, y_max = d["bounds"]
        bbox = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none')
        ax_2.add_patch(bbox)
        ax_2.text(x_min, y_min - 30, s=d["cls_name"] + ' ' + "1.0", color='white', verticalalignment='top',
                  bbox={'color': 'red', 'pad': 0})

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in img_det:
        print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

        # Rescale coordinates to original dimensions
        box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
        box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
        y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
        x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                 edgecolor=color,
                                 facecolor='none')
        ax_1.add_patch(bbox)
        ax_1.text(x1, y1 - 30, s=classes[int(cls_pred)] + ' ' + str('%.4f' % cls_conf.item()), color='white', verticalalignment='top',
                  bbox={'color': color, 'pad': 0})

    print(pad_x, pad_y)
    print(unpad_h, unpad_w)

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()

# print(f'Salient Variables {}')
# print(f'Detections {len(img_detections)}')
# print(f'Shapes should match {}')

# Save in file with parseable stuff...
