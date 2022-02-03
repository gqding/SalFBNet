from scipy import io
import numpy as np
import torch


def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor


def get_raw_fixations(fix_path):
    fix_data = io.loadmat(fix_path)
    fixations_array = [gaze[2] for gaze in fix_data['gaze'][:, 0]]
    return fixations_array, fix_data['resolution'].tolist()[0]


def process_raw_fixations(fixations_array, res):
    fix_map = np.zeros(res, dtype=np.uint8)
    for subject_fixations in fixations_array:
        fix_map[subject_fixations[:, 1] - 1, subject_fixations[:, 0] - 1] \
            = 255
    return fix_map


def get_salicon_fixation_map(fix_path):
    fixations_array, res = get_raw_fixations(fix_path)
    fix_map = process_raw_fixations(fixations_array, res)
    # cv2.imwrite(str(fix_map_file), fix_map)
    return fix_map


def get_cat2000_fixation_map(fix_path):
    fixations_array = io.loadmat(fix_path)
    fix_map = fixations_array['fixLocs'] * 255
    # cv2.imwrite(str(fix_map_file), fix_map)
    return fix_map
