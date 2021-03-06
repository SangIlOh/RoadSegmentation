﻿import numpy as np
import os
import time
import shutil
import struct
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

class DataSet( object):

    def __init__( self,
                 num_channel,
                 num_class,
                 num_class_wo_fake,
                 class_names,
                 class_colors,
                 dir,
                 weight_dir,
                 img_list,
                 label_list,
                 resize_shape,
                 normalize_val,
                 is_colormask = None):
        
        self._num_channel = num_channel
        self._num_class = num_class
        self._num_class_wo_fake = num_class_wo_fake
        self._class_names = class_names
        self._class_colors = class_colors

        self._dir = dir
        self._weight_dir = weight_dir
        self._img_list0 = np.array( img_list)
        self._img_list = np.array( img_list)
        self._label_list0 = np.array( label_list)
        self._label_list = np.array( label_list)
        self._resize_shape = resize_shape
        self._normalize_val = normalize_val
        self._is_colormask = is_colormask
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len( self._img_list)

        self._img_shape = None

    @property
    def dir( self):
        return self._dir

    @property
    def num_channel( self):
        return self._num_channel
    
    @property
    def num_class( self):
        return self._num_class

    @property
    def num_class_wo_fake( self):
        return self._num_class_wo_fake

    @property
    def class_names( self):
        return self._class_names

    @property
    def class_colors( self):
        return self._class_colors

    @property
    def img_list( self):
        return self._img_list

    @property
    def label_list( self):
        return self._label_list

    @property
    def num_examples( self):
        return self._num_examples

    @property
    def epochs_completed( self):
        return self._epochs_completed

    @property
    def img_shape( self):
        return self._img_shape

    @property
    def normalize_val( self):
        return self._normalize_val
    
    @property
    def images( self):
        batch_size = self._num_examples
        batch_image = np.ndarray( shape = ( ( batch_size,) + self._resize_shape + ( self._num_channel,)), dtype = np.float32)
        for nb in range( batch_size):
            img = self.img_read( self._dir + self._img_list[ nb], -1)
            img = img.astype( np.float32) / np.float32( self._normalize_val)
            if len( img.shape) == 2:
                img.shape = img.shape + ( 1,)
            batch_image[ nb, ...] = img
        return batch_image


    def resize( self, img, resize_shape):
        if resize_shape[ 0] == -1 and resize_shape[ 1] == -1:
            rimg = img
        elif resize_shape[ 0] == -1 and resize_shape[ 1] > 0:
            new_height = int( img.shape[ 0] * resize_shape[ 1] / img.shape[ 1])
            rimg = cv2.resize( img, ( resize_shape[ 1], new_height))
        elif resize_shape[ 0] > 0 and resize_shape[ 1] == -1:
            new_width = int( img.shape[ 1] * resize_shape[ 0] / img.shape[ 0])
            rimg = cv2.resize( img, ( new_width, resize_shape[ 0]))
        elif resize_shape[ 0] != img.shape[ 0] and resize_shape[ 1] != img.shape[ 1]:
            rimg = cv2.resize( img, ( resize_shape[ 1], resize_shape[ 0]))
        else:
            rimg = img

        rimg = cv2.resize( img, ( resize_shape[ 1], resize_shape[ 0]))
        return rimg


    def img_read( self, path, read_param = 0):
        img = cv2.imread( path, read_param)
        rimg = self.resize( img, self._resize_shape)
        return rimg


    def label_read( self, path, read_param = 0):
        img = cv2.imread( path, read_param)
        rimg = self.resize( img, self._resize_shape)
        return rimg

    def label_read_from_colormask(self, path, read_param = 0):
        img = cv2.imread( path, read_param)
        rimg = self.resize( img, self._resize_shape)

        label_imgs = []
        label0 = np.zeros( shape = ( 1,) + (rimg.shape[0], rimg.shape[1]) + ( self._num_class,), dtype = np.float32)

        for nc in range(self._num_class):
            mask_tmp = np.zeros((rimg.shape[0], rimg.shape[1]), np.uint8)
            logicb = np.uint8(rimg[:,:,0] == self._class_colors[nc][0])
            logicg = np.uint8(rimg[:,:,1] == self._class_colors[nc][1])
            logicr = np.uint8(rimg[:,:,2] == self._class_colors[nc][2])
            logic_cls = np.uint8(logicr+logicg+logicb)
            mask_tmp[logic_cls == 3] = 255

            label_imgs.append(mask_tmp)

            label_img = label_imgs[ nc]
            pixel_coords = np.where( label_img > 128)
            label0[ 0, pixel_coords[ 0], pixel_coords[ 1], nc] = 1

        background_pixel_coords = np.where( np.sum( label0, axis = 3) == 0)
        label0[ background_pixel_coords[ 0], background_pixel_coords[ 1], background_pixel_coords[ 2], -1] = 1
        
        return label0


    def next( self, batch_size = 1):

        start = self._index_in_epoch
        self._index_in_epoch += 1

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange( self._num_examples)
            np.random.shuffle( perm)
            self._img_list = self._img_list[ perm]
            self._label_list = self._label_list[ perm]
            start = 0
            self._index_in_epoch = 1
        end = self._index_in_epoch
                
        x0 = self.img_read( self._dir + self._img_list[ start], -1)
        
        label_imgs = []
        if self._is_colormask is None:
            for nc in range( self._num_class_wo_fake):
                label_imgs.append( self.label_read( self._dir + self._label_list[ start][ nc], 0))
            
        img_shape = x0.shape[ : 2]
        
        if len( x0.shape) == 2:
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape + ( 1,)
        else:
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape

        label0 = np.zeros( shape = ( 1,) + img_shape + ( self._num_class,), dtype = np.float32)
        if self._is_colormask is None:
            for nc in range( self._num_class_wo_fake):
                label_img = label_imgs[ nc]
                pixel_coords = np.where( label_img > 128)
                label0[ 0, pixel_coords[ 0], pixel_coords[ 1], nc] = 1
            background_pixel_coords = np.where( np.sum( label0, axis = 3) == 0)
            label0[ background_pixel_coords[ 0], background_pixel_coords[ 1], background_pixel_coords[ 2], -1] = 1
        else:
            label0 = self.label_read_from_colormask( self._dir + self._label_list[ start], 0)

        return x0, label0, self._img_list[ start ]


    def get( self, start, batch_size = 1):

        end = start + batch_size

        x0 = self.img_read( self._dir + self._img_list[ start], -1)
        
        label_imgs = []
        if self._is_colormask is None:
            for nc in range( self._num_class_wo_fake):
                label_imgs.append( self.label_read( self._dir + self._label_list[ start][ nc], 0))
            
        img_shape = x0.shape[ : 2]
        
        if len( x0.shape) == 2:
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape + ( 1,)
        else:
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape

        label0 = np.zeros( shape = ( 1,) + img_shape + ( self._num_class,), dtype = np.float32)
        if self._is_colormask is None:
            for nc in range( self._num_class_wo_fake):
                label_img = label_imgs[ nc]
                pixel_coords = np.where( label_img > 128)
                label0[ 0, pixel_coords[ 0], pixel_coords[ 1], nc] = 1
            background_pixel_coords = np.where( np.sum( label0, axis = 3) == 0)
            label0[ background_pixel_coords[ 0], background_pixel_coords[ 1], background_pixel_coords[ 2], -1] = 1
        else:
            label0 = self.label_read_from_colormask( self._dir + self._label_list[ start], 0)

        return x0, label0



    def save_prediction_img( self, save_path, img_name, batch_x, batch_y, batch_pr):

        batch_size = batch_y.shape[ 0]
        height = batch_y.shape[ 1]
        width = batch_y.shape[ 2]

        np_colors = np.array(self._class_colors)
        img_gt = np.uint8(np.argmax(batch_y, axis = 3))
        img_pred = np.uint8(np.argmax(batch_pr, axis = 3))
        img_input = np.uint8(np.argmax(batch_x, axis = 3))

        save_img = np.zeros((112*batch_size, 112 * 3, 3), np.uint8)
        batch_concat = np.zeros((batch_size, 112, 112*3, 3), np.uint8)
        for bn in range(batch_size):
            rgt = cv2.resize(img_gt[bn, :, :], (112, 112))
            rpred = cv2.resize(img_pred[bn, :, :], (112, 112))
            rinput = cv2.resize(img_input[bn, :, :], (112, 112))

            gt_res = np.zeros((112, 112, 3), np.uint8)
            pred_res = np.zeros((112, 112, 3), np.uint8)
            input_res = np.zeros((112, 112, 3), np.uint8)
            for yy in range(112):
                for xx in range(112):
                    gttmp = rgt[yy, xx]
                    gt_res[yy, xx, :] = np_colors[gttmp]

                    predtmp = rpred[yy, xx]
                    pred_res[yy, xx, :] = np_colors[predtmp]

                    inputtmp = rinput[yy, xx]
                    input_res[yy, xx, :] = np_colors[inputtmp]

            batch_concat[bn, :, :, :] = np.concatenate((input_res, gt_res, pred_res), axis = 1)
            #batch_concat[bn, :, :, :] = np.concatenate((np.uint8(cv2.resize(batch_x[bn, :, :], (112, 112))*255), gt_res, pred_res), axis = 1)

        save_img = np.concatenate((batch_concat[0, :, :, :], batch_concat[1, :, :, :], batch_concat[2, :, :, :], batch_concat[3, :, :, :]), axis = 0)
        cv2.imwrite( os.path.join( save_path, img_name + ".png"), save_img)



    def get_log( self):
        logging_str = [ "\t\t\tmodule : {0}\n".format( self.__module__),
                        "\t\t\tpath : {0}\n".format( self._dir),
                        "\t\t\tresize_shape : {0}\n".format( self._resize_shape),
                        "\t\t\tweight_dir : {0}\n".format( self._weight_dir if self._weight_dir != None else "None"),
                        "\t\t\tnormalize_val : {0}\n".format( self._normalize_val)]
        return ''.join( logging_str)


class DataSetReader( object):

    def __init__( self,
                 num_channel,
                 num_class,
                 num_class_wo_fake,
                 class_names,
                 class_colors,
                 dir,
                 is_colormask = None,
                 weight_dir = None,
                 resize_shape = ( -1, -1),
                 normalize_val = 255):
        self._num_channel = num_channel
        self._num_class = num_class
        self._num_class_wo_fake = num_class_wo_fake
        self._class_names = class_names
        self._class_colors = class_colors
        self._dir = dir
        self._is_colormask = is_colormask
        self._weight_dir = weight_dir
        self._resize_shape = resize_shape
        self._normalize_val = normalize_val


    def read_data_sets_from_flist( self, flist):

        if self._dir[ -1] != '/' and self._dir[ -1] != '\\':
            self._dir += '/'

        if self._weight_dir != None and self._weight_dir[ -1] != '/' and self._weight_dir[ -1] != '\\':
            self._weight_dir += '/'
        
        img_list = flist
        if self._is_colormask is None:
            label_list = [ [ os.path.splitext( img_path)[ 0] + "_" + str( nc + 1) + os.path.splitext( img_path)[ 1] for nc in range( self._num_class_wo_fake)] for img_path in img_list]
        else:
            label_list = [os.path.splitext( img_path)[ 0] + "_m" + os.path.splitext( img_path)[ 1] for img_path in img_list]

        return DataSet( self._num_channel,
                       self._num_class,
                       self._num_class_wo_fake,
                       self._class_names,
                       self._class_colors,
                       self._dir,
                       self._weight_dir,
                       img_list,
                       label_list,
                       self._resize_shape,
                       self._normalize_val,
                       self._is_colormask)