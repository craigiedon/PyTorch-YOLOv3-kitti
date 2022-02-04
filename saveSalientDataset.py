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
labels_folder = "/media/cinnes/Storage/Datasets/KITTI/ObjectDetection2D/Labels/training"
model_config_path = "config/yolov3-kitti.cfg"
img_path = '/media/cinnes/Storage/Datasets/KITTI/ObjectDetection2D/LeftCam/training'
img_size = 416
os.makdirs('salient_dataset', exist_ok=True)

model = Darknet(model_config_path, img_size=img_size)
model.load_weights(kitti_weights)
model.cuda()
model.eval()

dataloader = DataLoader(ImageFolder(img_path, img_size=img_size),
                        batch_size=1, shuffle=False, num_workers=8)

classes = load_classes('data/kitti.names')

imgs = []
img_detections = []

# Start in the label folder
# For each file in this folder
# Parse out the infos, load them into a big list
# In the "running through YOLO" part, lets compare the number of detections from YOLO to the number of true detections
# Do some sort of hungarian algorithm to match them up (write down how you will deal with false negatives / positives"
# Have two files: salient_variables.txt and predictions.txt. They should be exactly the same number of lines! (And it should be *more* lines than just the number of images)
salient_variable_dataset =

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    input_imgs = Variable(input_imgs.type(torch.cuda.FloatTensor))

    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, 0.8, 0.4)

        print(f'Batch {batch_i}')

        # imgs.extend(img_paths)
        img_detections.extend(detections)

print(f'Salient Variables {}')
print(f'Detections {len(img_detections)}')
print(f'Shapes should match {}')

# Save in file with parseable stuff...
