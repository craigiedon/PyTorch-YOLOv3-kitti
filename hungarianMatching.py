import numpy as np

from utils.utils import bbox_iou


def hungarian_matching(model_detections, tru_detections):
    # model/tru detections: [(cx, cy, w, h)]

    # Constraint: We can have false negatives, but we can't have false positives
    # I.e., if there are 4 true detections, and only 3 NN detections, we match the 3 best, and we assume the rest were false negatives (how do we represent these negatives?)
    # If there are 2 true detections and 5 NN Detections, pick the two best as matchups, and essentially just "discard" the false positives

    # Nothing to match, there are no true detections, and we are uninterested in false positives
    if len(tru_detections) == 0:
        return []

    # No model detections, so every tru detection is marked as a false negative
    if len(model_detections) == 0:
        return [(None, i) for i, _ in enumerate(tru_detections)]

    # Lets make an adjacency matrix of size len(model_detections) x len(tru_detections)
    adjacency_matrix = np.zeros(len(model_detections), len(tru_detections))
    for mi, model_det in enumerate(model_detections):
        for ti, tru_det in enumerate(tru_detections):
            adjacency_matrix[mi, ti] = bbox_iou(model_det, tru_det, False) # False required to state it is in cx, cy, w, h form...

    # Here's where we can actually do the hungarian algorithm


def run():
    model_detections = []
    tru_detections = []
    matches = hungarian_matching(model_detections, tru_detections)
    print(matches)


if __name__ == "__main__":
    # So...is it:
    # xmin, ymin, xmax, ymax? (The thing that comes out of the KITTI-Schema labels)

    # OR

    # cx, cy, width, height  (this comes out of the kitti2coco pipeline, its also what comes out of the NN detectors)
    # Lets do with this for the hungardian matchings part
    run()
