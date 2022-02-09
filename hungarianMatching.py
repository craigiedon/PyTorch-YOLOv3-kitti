def hungarian_matching():
    # Constraint: We can have false negatives, but we can't have false positives
    # I.e., if there are 4 true detections, and only 3 NN detections, we match the 3 best, and we assume the rest were false negatives (how do we represent these negatives?)
    # If there are 2 true detections and 5 NN Detections, pick the two best as matchups, and essentially just "discard" the false positives