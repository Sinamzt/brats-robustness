"""Independent sigmoid tumor regions converted to a consistent nested label map."""
import numpy as np


def probs_to_label_map(probs, thr=0.5):
    if probs.shape[0] != 3:
        raise ValueError("Expected TC, WT, ET probability channels")
    et = probs[2] > thr
    tc = (probs[0] > thr) | et
    wt = (probs[1] > thr) | tc
    out = np.zeros(probs.shape[1:], dtype=np.uint8)
    out[wt] = 2
    out[tc] = 1
    out[et] = 4
    return out


def gt_to_regions(seg):
    if not np.isin(seg, [0, 1, 2, 4]).all():
        raise ValueError("Expected BraTS2020 labels {0,1,2,4}")
    return {"WT": seg > 0, "TC": (seg == 1) | (seg == 4), "ET": seg == 4}


pred_to_regions = gt_to_regions
