import numpy as np
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
                 suffle_data,
                 border_type,
                 border_value,
                 resize_shape,
                 aug_class,
                 channelization,
                 normalize_val):
        
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
        self._border_type = border_type
        self._border_value = border_value
        self._resize_shape = resize_shape
        self._aug_class = aug_class
        self._channelization = channelization
        if self._channelization is not None:
            self._num_channel = self._channelization.num_channel
        self._normalize_val = normalize_val
                
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len( self._img_list)

        self._img_shape = None

        if suffle_data == True:
            perm = np.arange( self._num_examples)
            np.random.shuffle( perm)
            self._img_list = self._img_list[ perm]
            self._label_list = self._label_list[ perm]
            self._img_list0 = self._img_list0[ perm]
            self._label_list0 = self._label_list0[ perm]

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
            if self._aug_class != None:
                img = self._aug_class( img, self._normalize_val)
            if self._channelization != None:
                img = self._channelization( img, self._resize_shape, self._img_list[ nb])
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


    def next( self, pad_shape0, pad_shape1):

        batch_size = 1
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
        for nc in range( self._num_class_wo_fake):
            label_imgs.append( self.label_read( self._dir + self._label_list[ start][ nc], 0))
        if self._aug_class != None:
            x0, label_imgs = self._aug_class( x0, label_imgs, self._normalize_val)
        if self._channelization != None:
            x0 = self._channelization( x0)

        img_shape = x0.shape[ : 2]

        if len( x0.shape) == 2:

            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape + ( 1,)
        else:

            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape

        
        label0 = np.zeros( shape = ( 1,) + img_shape + ( self._num_class,), dtype = np.float32)
        for nc in range( self._num_class_wo_fake):
            label_img = label_imgs[ nc]
            pixel_coords = np.where( label_img > 128)
            label0[ 0, pixel_coords[ 0], pixel_coords[ 1], nc] = 1
        background_pixel_coords = np.where( np.sum( label0, axis = 3) == 0)
        label0[ background_pixel_coords[ 0], background_pixel_coords[ 1], background_pixel_coords[ 2], -1] = 1

        weight0 = np.full( fill_value = np.float32( 1), shape = ( 1,) + img_shape, dtype = np.float32)
        if self._weight_dir != None:
            weight_map = cv2.imread( self._weight_dir + self._img_list[ start][ : -4] + ".png", -1)
            weight_map = self.resize( weight_map, self._resize_shape)
            weight0[ 0, :, :] = weight_map

        return x0, label0, weight0, None, self._img_list[ start]


    def get( self, pad_shape0, pad_shape1, start):

        batch_size = 1
        end = start + 1

        x0 = self.img_read( self._dir + self._img_list0[ start], -1)
        label_imgs = []
        for nc in range( self._num_class_wo_fake):
            label_imgs.append( self.label_read( self._dir + self._label_list0[ start][ nc], 0))
        if self._aug_class != None:
            x0, label_imgs = self._aug_class( x0, label_imgs, self._normalize_val)
        if self._channelization != None:
            x0 = self._channelization( x0)

        img_shape = x0.shape[ : 2]

        if len( x0.shape) == 2:
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape + ( 1,)
        else:
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape
                    
        label0 = np.zeros( shape = ( 1,) + img_shape + ( self._num_class,), dtype = np.float32)
        for nc in range( self._num_class_wo_fake):
            label_img = label_imgs[ nc]
            pixel_coords = np.where( label_img > 128)
            label0[ 0, pixel_coords[ 0], pixel_coords[ 1], nc] = 1
        background_pixel_coords = np.where( np.sum( label0, axis = 3) == 0)
        label0[ background_pixel_coords[ 0], background_pixel_coords[ 1], background_pixel_coords[ 2], -1] = 1

        weight0 = np.full( fill_value = np.float32( 1), shape = ( 1,) + img_shape, dtype = np.float32)
        if self._weight_dir != None:
            weight_map = cv2.imread( self._weight_dir + self._img_list0[ start][ : -4] + ".png", -1)
            weight_map = self.resize( weight_map, self._resize_shape)
            weight0[ 0, :, :] = weight_map

        return x0, label0, weight0, None


    def get_window_batch( self, pad_shape0, pad_shape1, window_rect, start, batch_size):
        
        batch_x = np.ndarray( shape = ( ( batch_size,) + ( window_rect[ 2] + np.sum( pad_shape0), window_rect[ 3] + np.sum( pad_shape1), self._num_channel)), dtype = np.float32)
        batch_y = np.zeros( shape = ( ( batch_size,) + window_rect[ 2 :] + ( self._num_class,)), dtype = np.float32)
        batch_weight = np.full( shape = ( ( batch_size,) + window_rect[ 2 :]), fill_value = 1, dtype = np.float32)
        for nb in range( batch_size):
            x0, label0, weight0, _ = self.get( pad_shape0, pad_shape1, start + nb)
            batch_x[ nb, :, :, :]  = x0
            batch_y[ nb, :, :, :] = label0
            batch_weight[ nb, :, :] = weight0
        return batch_x, batch_y, batch_weight


    def insert_border_img( self, img, ih, iw, ir_cnt, iw_cnt, border_width, border_color):

        oimg = np.ndarray( shape = ( ih * ir_cnt + border_width * ( ir_cnt - 1),)
                                + ( iw * iw_cnt + border_width * ( iw_cnt - 1),) + ( 3,), dtype = np.uint8)
        
        for nb in range( ir_cnt):
            oimg_row_start = nb * ih + nb * border_width
            img_row_start = nb * ih
            for nc in range( iw_cnt):
                oimg_col_start = nc * iw + nc * border_width
                img_col_start = nc * iw
                oimg[ oimg_row_start : oimg_row_start + ih, oimg_col_start : oimg_col_start + iw, :] = img[ img_row_start : img_row_start + ih, img_col_start : img_col_start + iw, :]

        horizontal_border = np.full( shape = ( border_width, oimg.shape[ 1], 3), fill_value = border_color, dtype = np.uint8)
        for nb in range( 1, ir_cnt):
            oimg_row_start = nb * ih + ( nb - 1) * border_width
            oimg[ oimg_row_start : oimg_row_start + border_width, :, :] = horizontal_border
        vertical_border = np.full( shape = ( oimg.shape[ 0], border_width, 3), fill_value = border_color, dtype = np.uint8)
        for nc in range( 1, iw_cnt):
            oimg_col_start = nc * iw + ( nc - 1) * border_width
            oimg[ :, oimg_col_start : oimg_col_start + border_width, :] = vertical_border
        
        return oimg


    def save_prediction_img( self, save_path, img_name, batch_x, batch_y, batch_pr, save_img_type = 0, mask = None):

        if save_img_type == 2 and batch_y.shape[ 0] != 1:
            raise ValueError( "save_img_type == 2 and batch_y.shape[ 0] != 1")

        batch_size = batch_y.shape[ 0]
        height = batch_y.shape[ 1]
        width = batch_y.shape[ 2]
        pad0 = ( batch_x.shape[ 1] - batch_y.shape[ 1]) // 2
        pad1 = ( batch_x.shape[ 2] - batch_y.shape[ 2]) // 2

        if save_img_type != 5:
            if self._num_channel != 3:
                img_data = ( np.tile( batch_x.reshape( -1, width, 1), 3) * 255).astype( np.uint8)
            else:
                img_data = ( batch_x[ :, pad0 : -pad0, pad1 : -pad1, :].reshape( -1, width, 3) * 255).astype( np.uint8)
            np_colors = np.array( self._class_colors)
            img_gt = np_colors[ np.argmax( batch_y.reshape( -1, width, self._num_class), axis = 2)]
            img_pred = np_colors[ np.argmax( batch_pr.reshape( -1, width, self._num_class), axis = 2)]
            img_gt = img_gt.astype( np.uint8)
            img_pred = img_pred.astype( np.uint8)


            gt_background =  batch_y.reshape( -1, width, self._num_class)
            gt_background_pts = np.where( np.max( gt_background, axis = 2) == 0)
            img_gt[ gt_background_pts[ 0], gt_background_pts[ 1], :] = ( 0, 0, 0)

            gray_img_gt = cv2.cvtColor( img_gt, cv2.COLOR_BGR2GRAY)
            gray_img_pred = cv2.cvtColor( img_pred, cv2.COLOR_BGR2GRAY)
            and_gt_pred = np.logical_and( np.logical_and( img_gt[ :, :, 0] == img_pred[ :, :, 0], img_gt[ :, :, 1] == img_pred[ :, :, 1]), img_gt[ :, :, 2] == img_pred[ :, :, 2])

            dimg3 = ( gray_img_gt * ~and_gt_pred).astype( np.uint16) + ( gray_img_pred * ~and_gt_pred).astype( np.uint16)
            min_v = np.min( dimg3)
            max_v = np.max( dimg3)
            dimg3 = ( dimg3 - min_v) / ( max_v - min_v) * 255
            dimg3 = dimg3.astype( np.uint8)
            dimg3 = np.dstack( ( np.zeros_like( dimg3), np.zeros_like( dimg3), dimg3))

            simg3 = ( gray_img_gt * and_gt_pred).astype( np.uint16) + ( gray_img_pred * and_gt_pred).astype( np.uint16)
            min_v = np.min( simg3)
            max_v = np.max( simg3)
            if max_v - min_v > 0:
                simg3 = ( simg3 - min_v) / ( max_v - min_v) * 255
            else:
                simg3[ :, :] = 0
            simg3 = simg3.astype( np.uint8)
            simg3 = np.tile( simg3[ :, :, np.newaxis], ( 1, 1, 3))
            img3 = simg3 + dimg3
        
        
            if save_img_type == 2:
    
                nc = 0
                img_data0 = np.copy( img_data)
                colors = np.array( self._class_colors)
                gt_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_gt[ :, :, 0] == colors[ nc][ 0], img_gt[ :, :, 1] == colors[ nc][ 1]), img_gt[ :, :, 2] == colors[ nc][ 2])
                pr_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_pred[ :, :, 0] == colors[ nc][ 0], img_pred[ :, :, 1] == colors[ nc][ 1]), img_pred[ :, :, 2] == colors[ nc][ 2])
        
                gt_lines = np.where( np.uint8( 255) * np.logical_xor( gt_mask, cv2.erode( gt_mask, np.ones( ( 3, 3)))))
                img_data0[ gt_lines[ 0], gt_lines[ 1]] = ( 255, 0, 0)

                pr_lines = np.where( np.uint8( 255) * np.logical_xor( pr_mask, cv2.erode( pr_mask, np.ones( ( 3, 3)))))
                img_data0[ pr_lines[ 0], pr_lines[ 1]] = ( 0, 0, 255)

                nc = 1
                img_data1 = np.copy( img_data)
                colors = np.array( self._class_colors)
                gt_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_gt[ :, :, 0] == colors[ nc][ 0], img_gt[ :, :, 1] == colors[ nc][ 1]), img_gt[ :, :, 2] == colors[ nc][ 2])
                pr_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_pred[ :, :, 0] == colors[ nc][ 0], img_pred[ :, :, 1] == colors[ nc][ 1]), img_pred[ :, :, 2] == colors[ nc][ 2])
        
                gt_lines = np.where( np.uint8( 255) * np.logical_xor( gt_mask, cv2.erode( gt_mask, np.ones( ( 3, 3)))))
                img_data1[ gt_lines[ 0], gt_lines[ 1]] = ( 255, 0, 0)

                pr_lines = np.where( np.uint8( 255) * np.logical_xor( pr_mask, cv2.erode( pr_mask, np.ones( ( 3, 3)))))
                img_data1[ pr_lines[ 0], pr_lines[ 1]] = ( 0, 0, 255)

                pred_reshape = batch_pr.reshape( -1, width, self._num_class)
                pmaps = []
                for nc in range( self._num_class):
        
                    fig = plt.figure( frameon = False, figsize = ( width / 100, height / 100), dpi = 100)
                    #fig.set_size_inches( 1, 1)
                    ax = plt.Axes( fig, [ 0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes( ax)
                    ax.imshow( pred_reshape[ :, :, nc], cmap = plt.cm.jet, vmin = 0., vmax = 1.)
                    fig.canvas.draw()
                    jet_img = np.fromstring( fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
                    jet_img = jet_img.reshape( fig.canvas.get_width_height()[ : : -1] + ( 3,))
                    if jet_img.shape[ 0] != height or jet_img.shape[ 1] != width:
                        jet_img = cv2.resize( jet_img, ( width, height))
                    jet_img = cv2.cvtColor( jet_img, cv2.COLOR_BGR2RGB)
                    pmaps.append( jet_img)
                    plt.close()
                  
                img_r1 = np.concatenate( ( img_data0, img_data1), axis = 1)
                img_r2 = np.concatenate( ( pmaps[ 0], pmaps[ 1]), axis = 1)
                img = np.concatenate( ( img_r1, img_r2), axis = 0)
            
                #insert border
                img = self.insert_border_img( img, height, width, 2, 2, 10, ( 255, 255, 255))
            else:

                nc = 0
                colors = np.array( self._class_colors)
                gt_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_gt[ :, :, 0] == colors[ nc][ 0], img_gt[ :, :, 1] == colors[ nc][ 1]), img_gt[ :, :, 2] == colors[ nc][ 2])
                pr_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_pred[ :, :, 0] == colors[ nc][ 0], img_pred[ :, :, 1] == colors[ nc][ 1]), img_pred[ :, :, 2] == colors[ nc][ 2])
        
                gt_lines = np.where( np.uint8( 255) * np.logical_xor( gt_mask, cv2.erode( gt_mask, np.ones( ( 3, 3)))))
                img_data[ gt_lines[ 0], gt_lines[ 1]] = ( 255, 0, 0)

                pr_lines = np.where( np.uint8( 255) * np.logical_xor( pr_mask, cv2.erode( pr_mask, np.ones( ( 3, 3)))))
                img_data[ pr_lines[ 0], pr_lines[ 1]] = ( 0, 0, 255)

                
                pred_reshape = batch_pr.reshape( -1, width, self._num_class)
                pred_pmap = np.ndarray( shape = ( 300 * batch_size, 300 * self._num_class, 3), dtype = np.uint8)
                for nc in range( self._num_class):
        
                    fig = plt.figure( frameon = False)
                    fig.set_size_inches( 3, 3 * batch_size)
                    ax = plt.Axes( fig, [ 0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes( ax)
                    ax.imshow( pred_reshape[ :, :, nc], cmap = plt.cm.jet, vmin = 0., vmax = 1.)
                    fig.canvas.draw()
                    jet_img = np.fromstring( fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
                    jet_img = jet_img.reshape( fig.canvas.get_width_height()[ : : -1] + ( 3,))
                    pred_pmap[ :, nc * 300 : ( nc + 1) * 300, :] = jet_img
                    plt.close()
                pred_pmap = cv2.cvtColor( pred_pmap, cv2.COLOR_BGR2RGB)
                pred_pmap = self.insert_border_img( pred_pmap, 300, 300, batch_size, self._num_class, 10, ( 255, 255, 255))
            
                #img = np.concatenate( ( img_data, img_gt, img_pred, img3), axis = 1)
                #img = self.insert_border_img( img, height, width, batch_size, 4, 10, ( 255, 255, 255))
            
                img = np.concatenate( ( img_data, img_gt, img_pred, img3), axis = 1)
                img = self.insert_border_img( img, height, width, batch_size, 3, 10, ( 255, 255, 255))
            

            cv2.imwrite( os.path.join( save_path, img_name + ".png"), img)

            if save_img_type == 0:
                cv2.imwrite( os.path.join( save_path, img_name + "_pmap.png"), pred_pmap)

        else:
            np_colors = np.array(self._class_colors)
            img_gt = np.uint8(np.argmax(batch_y, axis = 3))
            img_pred = np.uint8(np.argmax(batch_pr, axis = 3))

            save_img = np.zeros((112*batch_size, 112 * 2, 3), np.uint8)
            batch_concat = np.zeros((batch_size, 112, 112*2, 3), np.uint8)
            for bn in range(batch_size):
                rgt = cv2.resize(img_gt[bn, :, :], (112, 112))
                rpred = cv2.resize(img_pred[bn, :, :], (112, 112))
                gt_res = np.zeros((112, 112, 3), np.uint8)
                pred_res = np.zeros((112, 112, 3), np.uint8)
                for yy in range(112):
                    for xx in range(112):
                        gttmp = rgt[yy, xx]
                        gt_res[yy, xx, :] = np_colors[gttmp]
                        predtmp = rpred[yy, xx]
                        pred_res[yy, xx, :] = np_colors[predtmp]

                batch_concat[bn, :, :, :] = np.concatenate((gt_res, pred_res), axis = 1)

            save_img = np.concatenate((batch_concat[0, :, :, :], batch_concat[1, :, :, :], batch_concat[2, :, :, :], batch_concat[3, :, :, :]), axis = 0)
            cv2.imwrite( os.path.join( save_path, img_name + ".png"), save_img)


    def get_log( self):
        logging_str = [ "\t\t\tmodule : {0}\n".format( self.__module__),
                        "\t\t\tpath : {0}\n".format( self._dir),
                        "\t\t\tresize_shape : {0}\n".format( self._resize_shape),
                        "\t\t\tchannelization : {0}\n".format( self._channelization.get_log() if self._channelization != None else "None"),
                        "\t\t\taug_class : {0}\n".format( self._aug_class.get_log() if self._aug_class != None else "None"),
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
                 weight_dir = None,
                 shuffle_data = False,
                 border_type = "constant",
                 border_value = 0,
                 resize_shape = ( -1, -1),
                 aug_class = None,
                 channelization = None,
                 normalize_val = 255):
        self._num_channel = num_channel
        self._num_class = num_class
        self._num_class_wo_fake = num_class_wo_fake
        self._class_names = class_names
        self._class_colors = class_colors
        self._dir = dir
        self._weight_dir = weight_dir
        self._shuffle_data = shuffle_data
        self._border_type = border_type
        self._border_value = border_value
        self._resize_shape = resize_shape
        self._aug_class = aug_class
        self._channelization = channelization
        self._normalize_val = normalize_val


    def read_data_sets( self):
    
        if self._dir[ -1] != '/' and self._dir[ -1] != '\\':
            self._dir += '/'

        if self._weight_dir != None and self._weight_dir[ -1] != '/' and self._weight_dir[ -1] != '\\':
            self._weight_dir += '/'

        flist = os.listdir( self._dir)
        img_list = flist
        for nc in range( self._num_class_wo_fake):
            img_list = list( filter( lambda path : "_" + str( nc + 1) + ".png" not in path, img_list))
    
        label_list = [ [ os.path.splitext( img_path)[ 0] + "_" + str( nc + 1) + os.path.splitext( img_path)[ 1] for nc in range( self._num_class_wo_fake)] for img_path in img_list]

        return DataSet( self._num_channel,
                       self._num_class,
                       self._num_class_wo_fake,
                       self._class_names,
                       self._class_colors,
                       self._dir,
                       self._weight_dir,
                       img_list,
                       label_list,
                       self._shuffle_data,
                       self._border_type,
                       self._border_value,
                       self._resize_shape,
                       self._aug_class,
                       self._channelization,
                       self._normalize_val)


    def read_data_sets_from_flist( self, flist):

        if self._dir[ -1] != '/' and self._dir[ -1] != '\\':
            self._dir += '/'

        if self._weight_dir != None and self._weight_dir[ -1] != '/' and self._weight_dir[ -1] != '\\':
            self._weight_dir += '/'
        
        img_list = flist
        label_list = [ [ os.path.splitext( img_path)[ 0] + "_" + str( nc + 1) + os.path.splitext( img_path)[ 1] for nc in range( self._num_class_wo_fake)] for img_path in img_list]

        return DataSet( self._num_channel,
                       self._num_class,
                       self._num_class_wo_fake,
                       self._class_names,
                       self._class_colors,
                       self._dir,
                       self._weight_dir,
                       img_list,
                       label_list,
                       self._shuffle_data,
                       self._border_type,
                       self._border_value,
                       self._resize_shape,
                       self._aug_class,
                       self._channelization,
                       self._normalize_val)