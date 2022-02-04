import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import patches
from matplotlib.ticker import NullLocator
import numpy as np


def parse_kitti_label(label_path):
    with open(label_path) as f:
        return list(filter(None, [parse_kitti_label_line(l) for l in f.readlines()]))


def parse_kitti_label_line(kitti_line):
    ws = kitti_line[:-1].split(' ')
    cls_name, vals = ws[0], [float(n) for n in ws[1:]]
    if cls_name != "DontCare":
        return {'cls_name': cls_name,
                'truncation': vals[0],
                'occlusion': vals[1],
                'alpha': vals[2],
                'bounds': np.array(vals[3:7]),
                'dim_3d': np.array(vals[7:10]),
                'loc_3d': np.array(vals[10:13]),
                'rot_y': vals[13]}
    return None


def parse_kitti_calibration(calibration_path):
    with open(calibration_path) as f:
        for l in f.readlines():
            # Rectified (Rotated) Camera coordinates to Image coordinates
            if l.startswith("P2:"):
                c_mat = np.array([float(n) for n in l[:-1].split(' ')[1:]]).reshape((3, 4))
            # Reference camera coordinates to Rectified camera coordinates
            if l.startswith("R0_rect:"):
                r0_mat = np.zeros((4, 4))
                r0_mat[0:3, 0:3] = np.array([float(n) for n in l[:-1].split(' ')[1:]]).reshape((3, 3))
                r0_mat[-1, -1] = 1.0

        return c_mat, r0_mat


def main():
    data_id = '000008'
    img = mpimg.imread(f"data/kitti/image/train/{data_id}.png")

    infos = parse_kitti_label(f"data/kitti/labels/train/{data_id}.txt")

    c_mat, r0_mat = parse_kitti_calibration(f"calibration/training/calib/{data_id}.txt")

    # c_mat[:, 2] = [0.0, 0.0, 1.0]
    print("c_mat", c_mat)

    fig, ax = plt.subplots(1)
    ax.axis('off')
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.imshow(img)

    for det_info in infos:
        loc_3d = det_info['loc_3d']
        print("loc_3d", loc_3d)
        projected_loc = c_mat @ np.append(loc_3d, 1.0)
        projected_loc = projected_loc / projected_loc[2]
        print("proj loc", projected_loc)

        ax.scatter(projected_loc[0], projected_loc[1], c='r')

        xmin, ymin, xmax, ymax = det_info['bounds']
        bbox = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                 edgecolor='blue',
                                 facecolor='none')
        ax.add_patch(bbox)

    plt.tight_layout()
    plt.show()
    # ax.text(x1, y1 - 30, s=cls_name, color='white',
    #          verticalalignment='top',
    #          bbox={'color': 'blue', 'pad': 0})


if __name__ == "__main__":
    main()
