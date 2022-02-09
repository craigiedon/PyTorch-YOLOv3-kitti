import os

img_folder = "/home/craig/data/KITTI/images/training"
img_paths = [os.path.join(img_folder, f_name) for f_name in sorted(os.listdir(img_folder))]


train_paths = img_paths[:len(img_paths) // 2]
val_paths = img_paths[(len(img_paths) // 2) + 1:]

with open("data/kitti/train.txt", 'w') as f:
    f.writelines([im_p + '\n' for im_p in train_paths])

with open("data/kitti/val.txt", 'w') as f:
    f.writelines([im_p + '\n' for im_p in val_paths])


