import numpy as np

import os
import shutil
import time
import cv2
import random
import tensorflow as tf
import math
from skimage import segmentation

from matplotlib import pyplot as plt

import sys

def ext_patch_list(img, gt_img, stride_step, patch_size, class_colors, OUTDIR, txtdir, imgidx, db_type):

    #patch_gt_list = np.zeros((patch_size[0], patch_size[1], len(class_colors)), np.uint8)
    patch_n = 0

    for yy in range(0, img.shape[0] - patch_size[0] - 1, stride_step[0]):
        for xx in range(0, img.shape[1] - patch_size[1] - 1, stride_step[1]):
            wr = [yy, xx, yy + patch_size[0], xx + patch_size[0]]

            patch_img = img[wr[0] : wr[2], wr[1] : wr[3], :]
            out_str = ''.join([OUTDIR, "_", str(patch_n), ".png"])
            cv2.imwrite(out_str, patch_img)
            with open(txtdir, 'a') as f_txt:
                written = ''.join([imgidx_str, "_", str(patch_n), ".png\n"])
                f_txt.write(written)

            patch_gt = gt_img[wr[0] : wr[2], wr[1] : wr[3], :]
            for gtn in range(len(class_colors)):
                masktmp = np.zeros(patch_size, np.uint8)
                logicb = np.uint8(patch_gt[:,:,0] == class_colors[gtn][2])
                logicg = np.uint8(patch_gt[:,:,1] == class_colors[gtn][1])
                logicr = np.uint8(patch_gt[:,:,2] == class_colors[gtn][0])
                logic_cls = np.uint8(logicr+logicg+logicb)
                masktmp[logic_cls == 3] = 255

                out_str = ''.join([OUTDIR, "_", str(patch_n), "_", str(gtn+1), ".png"])
                cv2.imwrite(out_str, masktmp)

            patch_n = patch_n + 1
            

if __name__ == "__main__":

    flist_base = "D:/Document/personal/EvLab-SSBenchmark/EvLab-SSBenchmark/"
    #db_type = ["train", "val"]
    db_type = ["val"]

    # RGB
    class_colors = [(128, 128, 128), (128, 0, 255), (192, 192, 128), (128, 64, 128), (60, 40, 222), (128, 128, 0), (192, 128, 128), ( 64, 64, 128), ( 64, 0, 128), ( 64, 64, 0), (0, 0, 0)]
    # Farmland, Garden, Woodland, Grassland, Building, Road, Structures, DiggingPile, Desert, Waters, Background
    num_class = 11

    ## patch params
    patch_size = (480, 480)
    stride_step = (400, 400)

    for n_db in range(len(db_type)):
        img_list = []
        gt_list = []
        txt_str = ''.join([flist_base, db_type[n_db], "_flist_ori.txt"])
        with open(txt_str, 'r') as f_db:
            for line in f_db:
                l = line.strip('.bmp\n')
                img_list.append(''.join([flist_base, db_type[n_db], "/", l, ".bmp"]))
                gt_list.append(''.join([flist_base, db_type[n_db], "/", l, "_2.tif"]))

        num_images = len(img_list)

        for now_img in range(num_images):
            img = cv2.imread(img_list[now_img], -1)
            gt_img = cv2.imread(gt_list[now_img], -1)

            if now_img is not 18 and db_type[n_db] == "train":
                ext_patch_list(img, gt_img, stride_step, patch_size, class_colors, ''.join([flist_base, db_type[n_db], "_patch_aug/", str(now_img + 1)]),
                               ''.join([flist_base, db_type[n_db], "_aug_patchlist.txt"]), now_img + 1, db_type[n_db])
            if now_img is not 0 and db_type[n_db] == "val":
                imgidx_str = img_list[now_img].strip(''.join([flist_base, db_type[n_db],".bmp"]))
                ext_patch_list(img, gt_img, stride_step, patch_size, class_colors, ''.join([flist_base, db_type[n_db], "_patch_aug/", imgidx_str]),
                               ''.join([flist_base, db_type[n_db], "_aug_patchlist.txt"]), imgidx_str, db_type[n_db])