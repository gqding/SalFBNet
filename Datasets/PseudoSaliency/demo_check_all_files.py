import os
import glob

import cv2
import numpy as np
from tqdm import tqdm

img_root = "./"
filename = os.path.join(img_root, "Train_List_PseudoSaliency.lst")
lines = [line.rstrip('\n') for line in open(filename)]
image_paths = list(map(lambda x: os.path.join(img_root, x.split(' ')[0]), lines))
label_paths = list(map(lambda x: os.path.join(img_root, x.split(' ')[1]), lines))
print("Starting to check all images and labels of training set...")
read_image=False
for img_path, lb_path in tqdm(zip(image_paths, label_paths), desc="training set", total=len(image_paths)):
    assert os.path.exists(img_path) and os.path.exists(lb_path)
    if read_image:
        image = cv2.imread(img_path)
        label = cv2.imread(lb_path, 0)
print("Finished! All files in training set are ready!")

filename_val = os.path.join(img_root, "Val_List_PseudoSaliency.lst")
lines_val = [line.rstrip('\n') for line in open(filename_val)]
image_paths_val = list(map(lambda x: os.path.join(img_root, x.split(' ')[0]), lines_val))
label_paths_val = list(map(lambda x: os.path.join(img_root, x.split(' ')[1]), lines_val))
print("Starting to check all images and labels of validation set...")
read_image=False
for img_path, lb_path in tqdm(zip(image_paths_val, label_paths_val), desc="validation set", total=len(image_paths_val)):
    assert os.path.exists(img_path) and os.path.exists(lb_path)
    if read_image:
        image = cv2.imread(img_path)
        label = cv2.imread(lb_path, 0)
print("Finished! All files in validation set are ready!")
