import numpy as np

from calib_project_vis import parse_kitti_label
from hungarianMatching import hungarian_matching
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


def run():
    kitti_weights = 'weights/yolov3-kitti.weights'
    model_config_path = "config/yolov3-kitti.cfg"
    img_folder = '/home/craig/data/KITTI/images/training'
    coco_labels_folder = "/home/craig/data/KITTI/labels/training"
    orig_salient_folder = "/home/craig/data/KITTI/labels-original/training"

    os.makedirs('salient_dataset', exist_ok=True)

    img_size = 416
    model = Darknet(model_config_path, img_size=img_size)
    model.load_weights(kitti_weights)
    model.cuda()
    model.eval()

    dataloader = DataLoader(ImageFolder(img_folder, img_size=img_size),
                            batch_size=4, shuffle=False, num_workers=16)

    classes = load_classes('data/kitti.names')

    img_filenames = sorted(os.listdir(img_folder))
    img_detections = torch.load("salient_dataset/detections.pt")

    label_paths = [os.path.join(coco_labels_folder, f_name) for f_name in sorted(os.listdir(coco_labels_folder))]

    tru_detections = [np.atleast_2d(np.loadtxt(lp)) for lp in label_paths]

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
        # print(img_filename)
        if len(img_det) == len(tru_det):
            continue

        img = np.array(Image.open(os.path.join(img_folder, img_filename)))
        fig, (ax_1, ax_2) = plt.subplots(2)
        ax_1.imshow(img)
        ax_2.imshow(img)

        if img_det is None:
            raise ValueError

        print(img_det.size())
        unique_labels = img_det[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        print("Class predictions:", n_cls_preds)
        bbox_colors = random.sample(colors, n_cls_preds)

        converted_img_det = np.array([rescale_nn_corners_to_orig_corners(imd[0], imd[1], imd[2], imd[3], img_size, img.shape[0], img.shape[1]) for imd in img_det])
        converted_tru_det = np.array([rescale_fractions_to_orig_corners(cx, cy, bw, bh, img.shape[0], img.shape[1]) for _, cx, cy, bw, bh in tru_det])

        matches = hungarian_matching(converted_img_det, converted_tru_det)

        print(matches)

        for i, (cls_num, *_) in enumerate(tru_det):
            x_min, y_min, x_max, y_max = converted_tru_det[i]
            bbox = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none')
            ax_2.add_patch(bbox)
            ax_2.text(x_min, y_min - 30, s=classes[int(cls_num)] + ' ' + "1.0", color='white', verticalalignment='top',
                      bbox={'color': 'red', 'pad': 0})

        for i, (_, _, _, _, conf, cls_conf, cls_pred) in enumerate(img_det):
            # print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            x1, y1, x2, y2 = converted_img_det[i]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            ax_1.add_patch(bbox)
            ax_1.text(x1, y1 - 30, s=classes[int(cls_pred)] + ' ' + str('%.4f' % cls_conf.item()), color='white', verticalalignment='top',
                      bbox={'color': color, 'pad': 0})

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.show()

    # print(f'Salient Variables {}')
    # print(f'Detections {len(img_detections)}')
    # print(f'Shapes should match {}')

    # Save in file with parseable stuff...


def rescale_nn_corners_to_orig_corners(x_min, y_min, x_max, y_max, nn_img_size, tru_img_h, tru_img_w):
    largest_dim_ratio = nn_img_size / max(tru_img_h, tru_img_w)

    # If height was largest, the x-dim would have been padded
    # If width was largest, the y-dim would have been padded
    pad_x = max(tru_img_h - tru_img_w, 0) * largest_dim_ratio
    pad_y = max(tru_img_w - tru_img_h, 0) * largest_dim_ratio

    unpad_h = nn_img_size - pad_y
    unpad_w = nn_img_size - pad_x

    box_h = int(((y_max - y_min) / unpad_h) * tru_img_h)
    box_w = int(((x_max - x_min) / unpad_w) * tru_img_w)
    y1 = int(((y_min - pad_y // 2) / unpad_h) * tru_img_h)
    x1 = int(((x_min - pad_x // 2) / unpad_w) * tru_img_w)
    return [x1, y1, x1 + box_w, y1 + box_h]


def rescale_fractions_to_orig_corners(cx, cy, box_width, box_height, tru_img_h, tru_img_w):
    x_min = (cx - box_width / 2) * tru_img_w
    y_min = (cy - box_height / 2) * tru_img_h
    x_max = (cx + box_width / 2) * tru_img_w
    y_max = (cy + box_height / 2) * tru_img_h

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


if __name__ == "__main__":
    run()
