import numpy as np
import tensorflow as tf
import os
import shutil
import time
import cv2
import time

import sys
lib_path = os.path.abspath( os.path.join( "..", "lib"))
sys.path.append( lib_path)
from MediVisionUnet import model_DeepLab2
from MediVisionUtil.dataset_rnet import DataSetReader

class channelization( object):
    
    @property
    def num_channel( self):
        return 3

    def gamma_correction( self, x, max_v, gamma):
        return ( max_v * ( ( x / max_v) ** ( 1 / gamma))).astype( x.dtype)

    def __call__( self, img):
    
        max_v = np.max( img)

        imgs = np.ndarray( shape = img.shape + ( 3,), dtype = img.dtype)
        imgs[ :, :, 0] = img
        imgs[ :, :, 1] = self.gamma_correction( img, max_v, 0.75)
        imgs[ :, :, 2] = self.gamma_correction( img, max_v, 1.5)

        return imgs

    def get_log( self):
        return "gamma_correction( 0.75, 1.0, 1.5)"

    
class func_save_conditonal_model( object):
    
    def __init__( self, class_idxs = [ 0, 0], save_path_names = [ "max_f1_c0", "max_r_c0__con_p_0_5"]):
        self._class_idxs = class_idxs
        self._save_path_names = save_path_names
        self._max = [ -1] * len( class_idxs)
            
    def __call__( self, epoch, cm):

        if epoch < 1:
            return []

        save = []
        if np.sum( cm[ 0, :]) == 0:
            r = 0
            p = 0
            f1 = 0
        elif np.sum( cm[ :, 0]) == 0:
            r = 0
            p = 0
            f1 = 0
        else:
            r = cm[ 0, 0] / np.sum( cm[ 0, :])
            p = cm[ 0, 0] / np.sum( cm[ :, 0])
            f1 = 2 * r * p / ( r + p)
        if f1 > self._max[ 0]:
            self._max[ 0] = f1
            save.append( self._save_path_names[ 0])
        if p >= 0.5 and r > self._max[ 1]:
            self._max[ 1] = r
            save.append( self._save_path_names[ 1])
        return save

    def get_log( self):
        return ""


if __name__ == "__main__":
    
    with open( "Bitewing_mt_2_flist/DS1_flist/train_patchlist.txt", "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        train_flist = ls
        
    with open( "Bitewing_mt_2_flist/DS1_flist/val_patchlist.txt", "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        test_flist = ls

    num_channel = 1
    num_class = 11
    num_class_wo_fake = 10
    class_names = [ "Farmland", "Garden", "Woodland", "Grassland", "Building", "Road", "Structures", "DiggingPile", "Desert", "Waters", "Background"]
    class_colors = [(128, 128, 128), (255, 0, 128), (128, 192, 192), (128, 64, 128), (222, 40, 60), (0, 128, 128), (128, 128, 192), ( 128, 64, 64), ( 128, 0, 64), ( 0, 64, 64), (0, 0, 0)]
    train_data = DataSetReader( num_channel,
                                num_class,
                                num_class_wo_fake,
                                class_names,
                                class_colors,
                                dir = "../dataset/mt_3/DS1/train_patch/",
                                shuffle_data = False,
                                resize_shape = ( 480,480),
                                border_type = "constant",
                                border_value = 0,
                                channelization = None,
                                normalize_val = 255).read_data_sets_from_flist( train_flist)

    test_data = DataSetReader( num_channel,
                                num_class,
                                num_class_wo_fake,
                                class_names,
                                class_colors,
                                dir = "../dataset/mt_3/DS1/val_patch/",
                                shuffle_data = False,
                                resize_shape = ( 480,480),
                                border_type = "constant",
                                border_value = 0,
                                channelization = None,
                                normalize_val = 255).read_data_sets_from_flist( test_flist)

    
    refinenet = model_DeepLab2.Refinenet( num_channel = 1,
                 num_class = num_class, 
                 output_HW = ( 480,480),
                 is_training= True)
    
    refinenet.train( train_data,
                "../models/deeplab/deeplab_0/",
                training_iters = 200,
                epochs = 150,
                display_step = 200,
                keep_prob = 1.0,
                opt_kwargs = { "cost": "dice_coefficient",
                              "optimizer": "adam",
                              "learning_rate": 1e-3,
                              "use_weight_map": False,
                              "batch_size": 4,
                              "pre_trained_model_iteration": None,
                              "test_data": test_data,
                              "save_model_epochs": [ 149],
                              "func_save_conditonal_model": func_save_conditonal_model()})