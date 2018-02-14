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
from MediVisionUnet import model_DeepLab2_bau2
from MediVisionUtil.dataset_rnet import DataSetReader
from MediVisionUnet import refinement_GAN

   
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
    
    ## set params
    TRAIN_DEEPLAB = False

    flist_base = "Bitewing_mt_2_flist/DS1_flist/"
    dltrain_flist = ''.join([flist_base, "dltrain_patchlist.txt"]) # flist for training DeepLab
    rftrain_flist = ''.join([flist_base, "rftrain_patchlist.txt"]) # flist for training refiner
    val_flist = ''.join([flist_base, "val_patchlist.txt"])

    datadir_base = "../dataset/refinegan/"
    train_datadir = ''.join([datadir_base, "train_patch/"])
    val_datadir = ''.join([datadir_base, "val_patch/"])

    deeplab_out_path = "../models/deeplab/deeplab_0/temp_bau3/"
    refiner_out_path = "../models/deeplab/deeplab_0/temp_refiner/"

    is_training = True
    num_channel = 3
    num_class = 11
    num_class_wo_fake = 10
    class_names = [ "Farmland", "Garden", "Woodland", "Grassland", "Building", "Road", "Structures", "DiggingPile", "Desert", "Waters", "Background"]
    class_colors = [(128, 128, 128), (255, 0, 128), (128, 192, 192), 
                    (128, 64, 128), (222, 40, 60), (0, 128, 128), 
                    (128, 128, 192), ( 128, 64, 64), ( 128, 0, 64), 
                    ( 0, 64, 64), (0, 0, 0)]

    target_shape = (480, 480)
    norm_val = 255
    batch_size = 3
    ###################################

    
        
    with open( val_flist, "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        test_flist = ls

    test_data = DataSetReader( num_channel,
                                num_class,
                                num_class_wo_fake,
                                class_names,
                                class_colors,
                                dir = val_datadir,
                                resize_shape = target_shape,
                                normalize_val = norm_val).read_data_sets_from_flist( test_flist)


    # train Deeplab
    if TRAIN_DEEPLAB:
        with open( dltrain_flist, "rt") as f:
            ls = f.readlines()
            ls = [ l.strip() for l in ls]
            train_flist = ls

        train_data = DataSetReader( num_channel,
                                num_class,
                                num_class_wo_fake,
                                class_names,
                                class_colors,
                                dir = train_datadir,
                                resize_shape = target_shape,
                                normalize_val = norm_val).read_data_sets_from_flist( train_flist)

        deeplab = model_DeepLab2_bau.DeepLabv2( num_channel = num_channel,
                     num_class = num_class, 
                     output_HW = target_shape,
                     is_training= is_training)
    
        deeplab.train( train_data,
                    deeplab_out_path,
                    training_iters = 2000,
                    epochs = 300,
                    display_step = 500,
                    keep_prob = 1.0,
                    opt_kwargs = { 
                                  "optimizer": "adam",
                                  "learning_rate": 1e-3,
                                  "use_weight_map": False,
                                  "batch_size": batch_size,
                                  "pre_trained_model_iteration": None,
                                  "test_data": test_data,
                                  "save_model_epochs": [ 0, 299],
                                  "func_save_conditonal_model": func_save_conditonal_model()})

        tf.reset_default_graph()


    for file in os.listdir(''.join([deeplab_out_path, "max_f1_c0"])):
        if file.endswith(".meta"):
            max_filename, trash = file.split(".meta")
    deeplab_premodel_meta = ''.join([deeplab_out_path, "max_f1_c0/", max_filename])

    # train refinenment
    with open( rftrain_flist, "rt") as f:
            ls = f.readlines()
            ls = [ l.strip() for l in ls]
            train_flist = ls

    train_data = DataSetReader( num_channel,
                            num_class,
                            num_class_wo_fake,
                            class_names,
                            class_colors,
                            dir = train_datadir,
                            resize_shape = target_shape,
                            normalize_val = norm_val).read_data_sets_from_flist( train_flist)

    num_images = train_data.num_examples

    refiner = refinement_GAN.build_refine_model(num_channel = num_channel,
                                                num_class = num_class,
                                                output_HW = target_shape,
                                                learning_rate = 0.0002)


    refiner.train(train_data,
                  refiner_out_path,
                  epochs = 300,
                  max_iter = 2000,
                  display_step = 500,
                  opt_kwargs = { "learning_rate": 0.0002,
                                  "batch_size": batch_size,
                                  "pre_trained_model_iteration": None,
                                  "test_data": test_data,
                                  "save_model_epochs": [ 0, 149, 299],
                                  "func_save_conditonal_model": func_save_conditonal_model(),
                                  "pre_gen_model": deeplab_premodel_meta})

