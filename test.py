import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time
import sys
import cv2


import model_DeepLab2_bau2
from dataset_rnet import DataSetReader

if __name__ == "__main__":

	img = cv2.imread("D:/Document/personal/EvLab-SSBenchmark/val_patch/39_0.png", -1)
	rimg = cv2.resize(img, (480, 480))
	nimg = np.float32(rimg) / 255.0
	nimg.shape = ( 1,) + nimg.shape

	class_names = [ "Farmland", "Garden", "Woodland", "Grassland", "Building", "Road", "Structures", "DiggingPile", "Desert", "Waters", "Background"]
	class_colors = [(128, 128, 128), (255, 0, 128), (128, 192, 192), (128, 64, 128), (222, 40, 60), (0, 128, 128), (128, 128, 192), ( 128, 64, 64), ( 128, 0, 64), ( 0, 64, 64), (0, 0, 0)]

	refinenet = model_DeepLab2_bau2.DeepLabv2( num_channel = 3,
                 num_class = len(class_names), 
                 output_HW = ( 480,480))

	result = refinenet.get_response(nimg, "C:/Users/sean.s.oh/Desktop/temp_bau3/max_f1_c0")

	np_colors = np.array(class_colors)
	img_pred = np.uint8(np.argmax(result[0, ...], axis = 2))
	pred_vis = np.zeros((img_pred.shape[0], img_pred.shape[1], 3), np.uint8)
	for yy in range(img_pred.shape[0]):
		for xx in range(img_pred.shape[1]):
			tmp = img_pred[yy, xx]
			pred_vis[yy, xx, :] = np_colors[tmp]

	cv2.imwrite("tmp.png", pred_vis)