import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time

log_formatter = logging.Formatter( "[%(asctime)s] %(message)s", datefmt = "%m-%d %H:%M:%S")
logger = logging.getLogger( "model_unet.py")
logger.setLevel( logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter( log_formatter)
logger.addHandler( console_handler)


def weight_variable( name, shape, mean = 0.0, stddev = 1.0):
    initial = tf.truncated_normal( shape = shape, dtype = tf.float32, mean = mean, stddev = stddev)
    return tf.Variable( initial, name = name)


def bias_variable( name, shape, value = 0.1):
    initial = tf.constant( value, shape = shape, dtype = tf.float32)
    return tf.Variable( initial, name = name)


def find_nth( haystack, needle, n):
        start = haystack.find( needle)
        while start >= 0 and n > 1:
            start = haystack.find( needle, start + len( needle))
            n -= 1
        return start

class DeepLabv2( object):
    
    _model_name = "deeplab_v2"

    def __init__( self, num_channel, num_class, output_HW, is_training = False, head_name_scope = "deeplab_v2", opt_kwargs = {}):
        
        if len( opt_kwargs):
            raise ValueError( "wrong opt_kwargs : %s" % ( str( opt_kwargs.keys())))

        with tf.name_scope( head_name_scope):
            self._num_channel = num_channel
            self._num_class = num_class
            self._output_HW = output_HW
            self._is_training = is_training

            input_HW = self._output_HW
            self._input_HW = input_HW

            with tf.name_scope( "input"):
                self._x = tf.placeholder( dtype = tf.float32, shape = [ None, input_HW[ 0], input_HW[ 1], self._num_channel], name = "x")
                self._y = tf.placeholder( dtype = tf.float32, shape = [ None, output_HW[ 0], output_HW[ 1], self._num_class], name = "y")
                self._keep_prob = tf.placeholder( dtype = tf.float32, name = "keep_prob")
            
            self._weights = []
            self._biases = []
        
            with tf.name_scope( "graph"):
                                
                with tf.name_scope( "layer0"):
                    w_conv1_0 = weight_variable( "W_conv1_0", shape = [ 7, 7, self._num_channel, 64], stddev = np.math.sqrt( 2.0 / ( 7 * 7 * self._num_channel)))
                    conv1 = tf.nn.conv2d( self._x, w_conv1_0, strides = [ 1, 2, 2, 1], padding = "VALID", name = "conv1")
                    bn_conv1 = tf.contrib.layers.batch_norm(conv1, is_training = is_training)
                    bn_conv1 = tf.nn.relu( bn_conv1, name = "bn_conv1")
                    pool1 = tf.nn.max_pool( bn_conv1, ksize = [ 1, 3, 3, 1], strides = [ 1, 2, 2, 1], padding = "VALID") # yn

                    w_conv2_0 = weight_variable( "W_conv2_0", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2a_branch1 = tf.nn.conv2d( pool1, w_conv2_0, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch1")
                    bn2a_branch1 = tf.contrib.layers.batch_norm(res2a_branch1, is_training = is_training)

                    self._weights.append( w_conv1_0)
                    self._weights.append( w_conv2_0)
                    
                with tf.name_scope( "layer1"):
                    w_conv1_1 = weight_variable( "W_conv1_1", shape = [ 1, 1, 64, 64], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2a_branch2a = tf.nn.conv2d( pool1, w_conv1_1, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch2a")
                    bn2a_branch2a = tf.contrib.layers.batch_norm(res2a_branch2a, is_training = is_training)
                    bn2a_branch2a = tf.nn.relu( bn2a_branch2a, name = "bn2a_branch2a")

                    w_conv2_1 = weight_variable( "W_conv2_1", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)))
                    res2a_branch2b = tf.nn.conv2d( bn2a_branch2a, w_conv2_1, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch2b")
                    bn2a_branch2b = tf.contrib.layers.batch_norm(res2a_branch2b, is_training = is_training)
                    bn2a_branch2b = tf.nn.relu( bn2a_branch2b, name = "bn2a_branch2b")

                    w_conv3_1 = weight_variable( "W_conv3_1", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2a_branch2c = tf.nn.conv2d( bn2a_branch2b, w_conv3_1, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch2c")
                    bn2a_branch2c = tf.contrib.layers.batch_norm(res2a_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_1)
                    self._weights.append( w_conv2_1)
                    self._weights.append( w_conv3_1)

                with tf.name_scope( "layer_2" ):
                    res2a = tf.add(bn2a_branch1, tf.image.resize_bilinear(bn2a_branch2c, [bn2a_branch1.get_shape()[1].value, bn2a_branch1.get_shape()[2].value,]), name = "res2a")
                    res2a_relu = tf.nn.relu( res2a, name = "res2a_relu")

                    w_conv1_2 = weight_variable( "W_conv1_2", shape = [ 1, 1, 256, 64], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res2b_branch2a = tf.nn.conv2d( res2a_relu, w_conv1_2, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2b_branch2a")
                    bn2b_branch2a = tf.contrib.layers.batch_norm(res2b_branch2a, is_training = is_training)
                    bn2b_branch2a = tf.nn.relu( bn2b_branch2a, name = "bn2b_branch2a")

                    w_conv2_2 = weight_variable( "W_conv2_2", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)))
                    res2b_branch2b = tf.nn.conv2d( bn2b_branch2a, w_conv2_2, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2b_branch2b")
                    bn2b_branch2b = tf.contrib.layers.batch_norm(res2b_branch2b, is_training = is_training)
                    bn2b_branch2b = tf.nn.relu( bn2b_branch2b, name = "bn2b_branch2b")

                    w_conv3_2 = weight_variable( "W_conv3_2", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2b_branch2c = tf.nn.conv2d( bn2b_branch2b, w_conv3_2, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2b_branch2c")
                    bn2b_branch2c = tf.contrib.layers.batch_norm(res2b_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_2)
                    self._weights.append( w_conv2_2)
                    self._weights.append( w_conv3_2)

                with tf.name_scope( "layer_3" ):
                    res2b = tf.add(res2a_relu, tf.image.resize_bilinear(bn2b_branch2c, [res2a_relu.get_shape()[1].value, res2a_relu.get_shape()[2].value,]), name = "res2b")
                    res2b_relu = tf.nn.relu( res2b, name = "res2b_relu")

                    w_conv1_3 = weight_variable( "W_conv1_3", shape = [ 1, 1, 256, 64], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res2c_branch2a = tf.nn.conv2d( res2b_relu, w_conv1_3, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2c_branch2a")
                    bn2c_branch2a = tf.contrib.layers.batch_norm(res2c_branch2a, is_training = is_training)
                    bn2c_branch2a = tf.nn.relu( bn2c_branch2a, name = "bn2c_branch2a")

                    w_conv2_3 = weight_variable( "W_conv2_3", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)))
                    res2c_branch2b = tf.nn.conv2d( bn2c_branch2a, w_conv2_3, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2c_branch2b")
                    bn2c_branch2b = tf.contrib.layers.batch_norm(res2c_branch2b, is_training = is_training)
                    bn2c_branch2b = tf.nn.relu( bn2c_branch2b, name = "bn2c_branch2b")

                    w_conv3_3 = weight_variable( "W_conv3_3", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2c_branch2c = tf.nn.conv2d( bn2c_branch2b, w_conv3_3, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2c_branch2c")
                    bn2c_branch2c = tf.contrib.layers.batch_norm(res2c_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_3)
                    self._weights.append( w_conv2_3)
                    self._weights.append( w_conv3_3)

                with tf.name_scope( "layer_4" ):
                    
                    res2c = tf.add(res2b_relu, tf.image.resize_bilinear(bn2c_branch2c, [res2b_relu.get_shape()[1].value, res2b_relu.get_shape()[2].value,]), name = "res2c")
                    res2c_relu = tf.nn.relu( res2c, name = "res2c_relu")

                    w_conv1_4 = weight_variable( "W_conv1_4", shape = [ 1, 1, 256, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res3a_branch1 = tf.nn.conv2d( res2c_relu, w_conv1_4, strides = [ 1, 2, 2, 1], padding = "VALID", name = "res3a_branch1")
                    bn3a_branch1 = tf.contrib.layers.batch_norm(res3a_branch1, is_training = is_training)

                    self._weights.append( w_conv1_4)

                with tf.name_scope( "layer_5" ):
                    w_conv1_5 = weight_variable( "W_conv1_5", shape = [ 1, 1, 256, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3a_branch2a = tf.nn.conv2d( res2c_relu, w_conv1_5, strides = [ 1, 2, 2, 1], padding = "VALID", name = "res3a_branch2a")
                    bn3a_branch2a = tf.contrib.layers.batch_norm(res3a_branch2a, is_training = is_training)
                    bn3a_branch2a = tf.nn.relu( bn3a_branch2a, name = "bn3a_branch2a")

                    w_conv2_5 = weight_variable( "W_conv2_5", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3a_branch2b = tf.nn.conv2d( bn3a_branch2a, w_conv2_5, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3a_branch2b")
                    bn3a_branch2b = tf.contrib.layers.batch_norm(res3a_branch2b, is_training = is_training)
                    bn3a_branch2b = tf.nn.relu( bn3a_branch2b, name = "bn3a_branch2b")

                    w_conv3_5 = weight_variable( "W_conv3_5", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3a_branch2c = tf.nn.conv2d( bn3a_branch2b, w_conv3_5, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3a_branch2c")
                    bn3a_branch2c = tf.contrib.layers.batch_norm(res3a_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_5)
                    self._weights.append( w_conv2_5)
                    self._weights.append( w_conv3_5)

                with tf.name_scope( "layer_6" ):
                    res3a = tf.add(bn3a_branch1, tf.image.resize_bilinear(bn3a_branch2c, [bn3a_branch1.get_shape()[1].value, bn3a_branch1.get_shape()[2].value,]), name = "res3a")
                    res3a_relu = tf.nn.relu( res3a, name = "res3a_relu")

                    w_conv1_6 = weight_variable( "W_conv1_6", shape = [ 1, 1, 512, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3b1_branch2a = tf.nn.conv2d( res3a_relu, w_conv1_6, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b1_branch2a")
                    bn3b1_branch2a = tf.contrib.layers.batch_norm(res3b1_branch2a, is_training = is_training)
                    bn3b1_branch2a = tf.nn.relu( bn3b1_branch2a, name = "bn3b1_branch2a")

                    w_conv2_6 = weight_variable( "W_conv2_6", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3b1_branch2b = tf.nn.conv2d( bn3b1_branch2a, w_conv2_6, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b1_branch2b")
                    bn3b1_branch2b = tf.contrib.layers.batch_norm(res3b1_branch2b, is_training = is_training)
                    bn3b1_branch2b = tf.nn.relu( bn3b1_branch2b, name = "bn3b1_branch2b")

                    w_conv3_6 = weight_variable( "W_conv3_6", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3b1_branch2c = tf.nn.conv2d( bn3b1_branch2b, w_conv3_6, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b1_branch2c")
                    bn3b1_branch2c = tf.contrib.layers.batch_norm(res3b1_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_6)
                    self._weights.append( w_conv2_6)
                    self._weights.append( w_conv3_6)

                with tf.name_scope( "layer_7" ):
                    res3b1 = tf.add(res3a_relu, tf.image.resize_bilinear(bn3b1_branch2c, [res3a_relu.get_shape()[1].value, res3a_relu.get_shape()[2].value,]), name = "res3b1")
                    res3b1_relu = tf.nn.relu( res3b1, name = "res3b1_relu")

                    w_conv1_7 = weight_variable( "W_conv1_7", shape = [ 1, 1, 512, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3b2_branch2a = tf.nn.conv2d( res3b1_relu, w_conv1_7, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b2_branch2a")
                    bn3b2_branch2a = tf.contrib.layers.batch_norm(res3b2_branch2a, is_training = is_training)
                    bn3b2_branch2a = tf.nn.relu( bn3b2_branch2a, name = "bn3b2_branch2a")

                    w_conv2_7 = weight_variable( "W_conv2_7", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3b2_branch2b = tf.nn.conv2d( bn3b2_branch2a, w_conv2_7, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b2_branch2b")
                    bn3b2_branch2b = tf.contrib.layers.batch_norm(res3b2_branch2b, is_training = is_training)
                    bn3b2_branch2b = tf.nn.relu( bn3b1_branch2b, name = "bn3b2_branch2b")

                    w_conv3_7 = weight_variable( "W_conv3_7", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3b2_branch2c = tf.nn.conv2d( bn3b2_branch2b, w_conv3_7, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b2_branch2c")
                    bn3b2_branch2c = tf.contrib.layers.batch_norm(res3b2_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_7)
                    self._weights.append( w_conv2_7)
                    self._weights.append( w_conv3_7)

                with tf.name_scope( "layer_8" ):
                    res3b2 = tf.add(res3b1_relu, tf.image.resize_bilinear(bn3b2_branch2c, [res3b1_relu.get_shape()[1].value, res3b1_relu.get_shape()[2].value,]), name = "res3b2")
                    res3b2_relu = tf.nn.relu( res3b2, name = "res3b2_relu")

                    w_conv1_8 = weight_variable( "W_conv1_8", shape = [ 1, 1, 512, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3b3_branch2a = tf.nn.conv2d( res3b2_relu, w_conv1_8, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b3_branch2a")
                    bn3b3_branch2a = tf.contrib.layers.batch_norm(res3b3_branch2a, is_training = is_training)
                    bn3b3_branch2a = tf.nn.relu( bn3b2_branch2a, name = "bn3b3_branch2a")

                    w_conv2_8 = weight_variable( "W_conv2_8", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3b3_branch2b = tf.nn.conv2d( bn3b3_branch2a, w_conv2_8, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b3_branch2b")
                    bn3b3_branch2b = tf.contrib.layers.batch_norm(res3b3_branch2b, is_training = is_training)
                    bn3b3_branch2b = tf.nn.relu( bn3b3_branch2b, name = "bn3b3_branch2b")

                    w_conv3_8 = weight_variable( "W_conv3_8", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3b3_branch2c = tf.nn.conv2d( bn3b3_branch2b, w_conv3_8, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b3_branch2c")
                    bn3b3_branch2c = tf.contrib.layers.batch_norm(res3b3_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_8)
                    self._weights.append( w_conv2_8)
                    self._weights.append( w_conv3_8)

                with tf.name_scope( "layer_9" ):
                    res3b3 = tf.add(res3b2_relu, tf.image.resize_bilinear(bn3b3_branch2c, [res3b2_relu.get_shape()[1].value, res3b2_relu.get_shape()[2].value,]), name = "res3b3")
                    res3b3_relu = tf.nn.relu( res3b3, name = "res3b3_relu")

                    w_conv1_9 = weight_variable( "W_conv1_9", shape = [ 1, 1, 512, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res4a_branch1 = tf.nn.conv2d( res3b3_relu, w_conv1_9, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4a_branch1")
                    bn4a_branch1 = tf.contrib.layers.batch_norm(res4a_branch1, is_training = is_training)

                    self._weights.append( w_conv1_9)

                with tf.name_scope( "layer_10" ):
                    w_conv1_10 = weight_variable( "W_conv1_10", shape = [ 1, 1, 512, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res4a_branch2a = tf.nn.conv2d( res3b3_relu, w_conv1_10, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4a_branch2a")
                    bn4a_branch2a = tf.contrib.layers.batch_norm(res4a_branch2a, is_training = is_training)
                    bn4a_branch2a = tf.nn.relu( bn4a_branch2a, name = "bn4a_branch2a")

                    w_conv2_10 = weight_variable( "W_conv2_10", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4a_branch2b = tf.nn.atrous_conv2d( bn4a_branch2a, w_conv2_10, rate = 2, padding = "SAME", name = "res4a_branch2b")
                    bn4a_branch2b = tf.contrib.layers.batch_norm(res4a_branch2b, is_training = is_training)
                    bn4a_branch2b = tf.nn.relu( bn4a_branch2b, name = "bn4a_branch2b")

                    w_conv3_10 = weight_variable( "W_conv3_10", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4a_branch2c = tf.nn.conv2d( bn4a_branch2b, w_conv3_10, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4a_branch2c")
                    bn4a_branch2c = tf.contrib.layers.batch_norm(res4a_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_10)
                    self._weights.append( w_conv2_10)
                    self._weights.append( w_conv3_10)

                with tf.name_scope( "layer_11" ):
                    res4a = tf.add(bn4a_branch1, tf.image.resize_bilinear(bn4a_branch2c, [bn4a_branch1.get_shape()[1].value, bn4a_branch1.get_shape()[2].value,]), name = "res4a")
                    res4a_relu = tf.nn.relu( res4a, name = "res4a_relu")

                    w_conv1_11 = weight_variable( "W_conv1_11", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b1_branch2a = tf.nn.conv2d( res4a_relu, w_conv1_11, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b1_branch2a")
                    bn4b1_branch2a = tf.contrib.layers.batch_norm(res4b1_branch2a, is_training = is_training)
                    bn4b1_branch2a = tf.nn.relu( bn4b1_branch2a, name = "bn4b1_branch2a")

                    w_conv2_11 = weight_variable( "W_conv2_11", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b1_branch2b = tf.nn.atrous_conv2d( bn4b1_branch2a, w_conv2_11, rate = 2, padding = "SAME", name = "res4b1_branch2b")
                    bn4b1_branch2b = tf.contrib.layers.batch_norm(res4b1_branch2b, is_training = is_training)
                    bn4b1_branch2b = tf.nn.relu( bn4b1_branch2b, name = "bn4b1_branch2b")

                    w_conv3_11 = weight_variable( "W_conv3_11", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b1_branch2c = tf.nn.conv2d( bn4b1_branch2b, w_conv3_11, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b1_branch2c")
                    bn4b1_branch2c = tf.contrib.layers.batch_norm(res4b1_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_11)
                    self._weights.append( w_conv2_11)
                    self._weights.append( w_conv3_11)

                with tf.name_scope( "layer_12" ):
                    res4b1 = tf.add(res4a_relu, tf.image.resize_bilinear(bn4b1_branch2c, [res4a_relu.get_shape()[1].value, res4a_relu.get_shape()[2].value,]), name = "res4b1")
                    res4b1_relu = tf.nn.relu( res4b1, name = "res4b1_relu")

                    w_conv1_12 = weight_variable( "W_conv1_12", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b2_branch2a = tf.nn.conv2d( res4b1_relu, w_conv1_12, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b2_branch2a")
                    bn4b2_branch2a = tf.contrib.layers.batch_norm(res4b2_branch2a, is_training = is_training)
                    bn4b2_branch2a = tf.nn.relu( bn4b1_branch2a, name = "bn4b2_branch2a")

                    w_conv2_12 = weight_variable( "W_conv2_12", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b2_branch2b = tf.nn.atrous_conv2d( bn4b2_branch2a, w_conv2_12, rate = 2, padding = "SAME", name = "res4b2_branch2b")
                    bn4b2_branch2b = tf.contrib.layers.batch_norm(res4b2_branch2b, is_training = is_training)
                    bn4b2_branch2b = tf.nn.relu( bn4b2_branch2b, name = "bn4b2_branch2b")

                    w_conv3_12 = weight_variable( "W_conv3_12", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b2_branch2c = tf.nn.conv2d( bn4b2_branch2b, w_conv3_12, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b2_branch2c")
                    bn4b2_branch2c = tf.contrib.layers.batch_norm(res4b2_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_12)
                    self._weights.append( w_conv2_12)
                    self._weights.append( w_conv3_12)

                with tf.name_scope( "layer_13" ):
                    res4b2 = tf.add(res4b1_relu, tf.image.resize_bilinear(bn4b2_branch2c, [res4b1_relu.get_shape()[1].value, res4b1_relu.get_shape()[2].value,]), name = "res4b2")
                    res4b2_relu = tf.nn.relu( res4b2, name = "res4b2_relu")

                    w_conv1_13 = weight_variable( "W_conv1_13", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b3_branch2a = tf.nn.conv2d( res4b2_relu, w_conv1_13, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b3_branch2a")
                    bn4b3_branch2a = tf.contrib.layers.batch_norm(res4b3_branch2a, is_training = is_training)
                    bn4b3_branch2a = tf.nn.relu( bn4b3_branch2a, name = "bn4b3_branch2a")

                    w_conv2_13 = weight_variable( "W_conv2_13", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b3_branch2b = tf.nn.atrous_conv2d( bn4b3_branch2a, w_conv2_13, rate = 2, padding = "SAME", name = "res4b3_branch2b")
                    bn4b3_branch2b = tf.contrib.layers.batch_norm(res4b3_branch2b, is_training = is_training)
                    bn4b3_branch2b = tf.nn.relu( bn4b3_branch2b, name = "bn4b3_branch2b")

                    w_conv3_13 = weight_variable( "W_conv3_13", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b3_branch2c = tf.nn.conv2d( bn4b3_branch2b, w_conv3_13, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b3_branch2c")
                    bn4b3_branch2c = tf.contrib.layers.batch_norm(res4b3_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_13)
                    self._weights.append( w_conv2_13)
                    self._weights.append( w_conv3_13)

                with tf.name_scope( "layer_14" ):
                    res4b3 = tf.add(res4b2_relu, tf.image.resize_bilinear(bn4b3_branch2c, [res4b2_relu.get_shape()[1].value, res4b2_relu.get_shape()[2].value,]), name = "res4b3")
                    res4b3_relu = tf.nn.relu( res4b3, name = "res4b3_relu")

                    w_conv1_14 = weight_variable( "W_conv1_14", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b4_branch2a = tf.nn.conv2d( res4b3_relu, w_conv1_14, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b4_branch2a")
                    bn4b4_branch2a = tf.contrib.layers.batch_norm(res4b4_branch2a, is_training = is_training)
                    bn4b4_branch2a = tf.nn.relu( bn4b4_branch2a, name = "bn4b4_branch2a")

                    w_conv2_14 = weight_variable( "W_conv2_14", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b4_branch2b = tf.nn.atrous_conv2d( bn4b4_branch2a, w_conv2_14, rate = 2, padding = "SAME", name = "res4b4_branch2b")
                    bn4b4_branch2b = tf.contrib.layers.batch_norm(res4b4_branch2b, is_training = is_training)
                    bn4b4_branch2b = tf.nn.relu( bn4b4_branch2b, name = "bn4b4_branch2b")

                    w_conv3_14 = weight_variable( "W_conv3_14", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b4_branch2c = tf.nn.conv2d( bn4b4_branch2b, w_conv3_14, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b4_branch2c")
                    bn4b4_branch2c = tf.contrib.layers.batch_norm(res4b4_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_14)
                    self._weights.append( w_conv2_14)
                    self._weights.append( w_conv3_14)

                with tf.name_scope( "layer_15" ):
                    res4b4 = tf.add(res4b3_relu, tf.image.resize_bilinear(bn4b4_branch2c, [res4b3_relu.get_shape()[1].value, res4b3_relu.get_shape()[2].value,]), name = "res4b4")
                    res4b4_relu = tf.nn.relu( res4b4, name = "res4b4_relu")

                    w_conv1_15 = weight_variable( "W_conv1_15", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b5_branch2a = tf.nn.conv2d( res4b4_relu, w_conv1_15, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b5_branch2a")
                    bn4b5_branch2a = tf.contrib.layers.batch_norm(res4b5_branch2a, is_training = is_training)
                    bn4b5_branch2a = tf.nn.relu( bn4b5_branch2a, name = "bn4b5_branch2a")

                    w_conv2_15 = weight_variable( "W_conv2_15", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b5_branch2b = tf.nn.atrous_conv2d( bn4b5_branch2a, w_conv2_15, rate = 2, padding = "SAME", name = "res4b5_branch2b")
                    bn4b5_branch2b = tf.contrib.layers.batch_norm(res4b5_branch2b, is_training = is_training)
                    bn4b5_branch2b = tf.nn.relu( bn4b5_branch2b, name = "bn4b5_branch2b")

                    w_conv3_15 = weight_variable( "W_conv3_15", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b5_branch2c = tf.nn.conv2d( bn4b5_branch2b, w_conv3_15, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b5_branch2c")
                    bn4b5_branch2c = tf.contrib.layers.batch_norm(res4b5_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_15)
                    self._weights.append( w_conv2_15)
                    self._weights.append( w_conv3_15)

                with tf.name_scope( "layer_16" ):
                    res4b5 = tf.add(res4b4_relu, tf.image.resize_bilinear(bn4b5_branch2c, [res4b4_relu.get_shape()[1].value, res4b4_relu.get_shape()[2].value,]), name = "res4b5")
                    res4b5_relu = tf.nn.relu( res4b5, name = "res4b5_relu")

                    w_conv1_16 = weight_variable( "W_conv1_16", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b6_branch2a = tf.nn.conv2d( res4b5_relu, w_conv1_16, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b6_branch2a")
                    bn4b6_branch2a = tf.contrib.layers.batch_norm(res4b6_branch2a, is_training = is_training)
                    bn4b6_branch2a = tf.nn.relu( bn4b6_branch2a, name = "bn4b6_branch2a")

                    w_conv2_16 = weight_variable( "W_conv2_16", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b6_branch2b = tf.nn.atrous_conv2d( bn4b6_branch2a, w_conv2_16, rate = 2, padding = "SAME", name = "res4b6_branch2b")
                    bn4b6_branch2b = tf.contrib.layers.batch_norm(res4b6_branch2b, is_training = is_training)
                    bn4b6_branch2b = tf.nn.relu( bn4b6_branch2b, name = "bn4b6_branch2b")

                    w_conv3_16 = weight_variable( "W_conv3_16", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b6_branch2c = tf.nn.conv2d( bn4b6_branch2b, w_conv3_16, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b6_branch2c")
                    bn4b6_branch2c = tf.contrib.layers.batch_norm(res4b6_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_16)
                    self._weights.append( w_conv2_16)
                    self._weights.append( w_conv3_16)

                with tf.name_scope( "layer_17" ):
                    res4b6 = tf.add(res4b5_relu, tf.image.resize_bilinear(bn4b6_branch2c, [res4b5_relu.get_shape()[1].value, res4b5_relu.get_shape()[2].value,]), name = "res4b6")
                    res4b6_relu = tf.nn.relu( res4b6, name = "res4b6_relu")

                    w_conv1_17 = weight_variable( "W_conv1_17", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b7_branch2a = tf.nn.conv2d( res4b6_relu, w_conv1_17, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b7_branch2a")
                    bn4b7_branch2a = tf.contrib.layers.batch_norm(res4b7_branch2a, is_training = is_training)
                    bn4b7_branch2a = tf.nn.relu( bn4b7_branch2a, name = "bn4b7_branch2a")

                    w_conv2_17 = weight_variable( "W_conv2_17", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b7_branch2b = tf.nn.atrous_conv2d( bn4b7_branch2a, w_conv2_17, rate = 2, padding = "SAME", name = "res4b7_branch2b")
                    bn4b7_branch2b = tf.contrib.layers.batch_norm(res4b7_branch2b, is_training = is_training)
                    bn4b7_branch2b = tf.nn.relu( bn4b7_branch2b, name = "bn4b7_branch2b")

                    w_conv3_17 = weight_variable( "W_conv3_17", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b7_branch2c = tf.nn.conv2d( bn4b7_branch2b, w_conv3_17, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b7_branch2c")
                    bn4b7_branch2c = tf.contrib.layers.batch_norm(res4b7_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_17)
                    self._weights.append( w_conv2_17)
                    self._weights.append( w_conv3_17)

                with tf.name_scope( "layer_18" ):
                    res4b7 = tf.add(res4b6_relu, tf.image.resize_bilinear(bn4b7_branch2c, [res4b6_relu.get_shape()[1].value, res4b6_relu.get_shape()[2].value,]), name = "res4b7")
                    res4b7_relu = tf.nn.relu( res4b7, name = "res4b7_relu")

                    w_conv1_18 = weight_variable( "W_conv1_18", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b8_branch2a = tf.nn.conv2d( res4b7_relu, w_conv1_18, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b8_branch2a")
                    bn4b8_branch2a = tf.contrib.layers.batch_norm(res4b8_branch2a, is_training = is_training)
                    bn4b8_branch2a = tf.nn.relu( bn4b8_branch2a, name = "bn4b8_branch2a")

                    w_conv2_18 = weight_variable( "W_conv2_18", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b8_branch2b = tf.nn.atrous_conv2d( bn4b8_branch2a, w_conv2_18, rate = 2, padding = "SAME", name = "res4b8_branch2b")
                    bn4b8_branch2b = tf.contrib.layers.batch_norm(res4b8_branch2b, is_training = is_training)
                    bn4b8_branch2b = tf.nn.relu( bn4b8_branch2b, name = "bn4b8_branch2b")

                    w_conv3_18 = weight_variable( "W_conv3_18", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b8_branch2c = tf.nn.conv2d( bn4b8_branch2b, w_conv3_18, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b8_branch2c")
                    bn4b8_branch2c = tf.contrib.layers.batch_norm(res4b8_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_18)
                    self._weights.append( w_conv2_18)
                    self._weights.append( w_conv3_18)

                with tf.name_scope( "layer_19" ):
                    res4b8 = tf.add(res4b7_relu, tf.image.resize_bilinear(bn4b8_branch2c, [res4b7_relu.get_shape()[1].value, res4b7_relu.get_shape()[2].value,]), name = "res4b8")
                    res4b8_relu = tf.nn.relu( res4b8, name = "res4b8_relu")

                    w_conv1_19 = weight_variable( "W_conv1_19", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b9_branch2a = tf.nn.conv2d( res4b8_relu, w_conv1_19, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b9_branch2a")
                    bn4b9_branch2a = tf.contrib.layers.batch_norm(res4b9_branch2a, is_training = is_training)
                    bn4b9_branch2a = tf.nn.relu( bn4b9_branch2a, name = "bn4b9_branch2a")

                    w_conv2_19 = weight_variable( "W_conv2_19", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b9_branch2b = tf.nn.atrous_conv2d( bn4b9_branch2a, w_conv2_19, rate = 2, padding = "SAME", name = "res4b9_branch2b")
                    bn4b9_branch2b = tf.contrib.layers.batch_norm(res4b9_branch2b, is_training = is_training)
                    bn4b9_branch2b = tf.nn.relu( bn4b9_branch2b, name = "bn4b9_branch2b")

                    w_conv3_19 = weight_variable( "W_conv3_19", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b9_branch2c = tf.nn.conv2d( bn4b9_branch2b, w_conv3_19, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b9_branch2c")
                    bn4b9_branch2c = tf.contrib.layers.batch_norm(res4b9_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_19)
                    self._weights.append( w_conv2_19)
                    self._weights.append( w_conv3_19)

                with tf.name_scope( "layer_20" ):
                    res4b9 = tf.add(res4b8_relu, tf.image.resize_bilinear(bn4b9_branch2c, [res4b8_relu.get_shape()[1].value, res4b8_relu.get_shape()[2].value,]), name = "res4b9")
                    res4b9_relu = tf.nn.relu( res4b9, name = "res4b9_relu")

                    w_conv1_20 = weight_variable( "W_conv1_20", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b10_branch2a = tf.nn.conv2d( res4b9_relu, w_conv1_20, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b10_branch2a")
                    bn4b10_branch2a = tf.contrib.layers.batch_norm(res4b10_branch2a, is_training = is_training)
                    bn4b10_branch2a = tf.nn.relu( bn4b10_branch2a, name = "bn4b10_branch2a")

                    w_conv2_20 = weight_variable( "W_conv2_20", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b10_branch2b = tf.nn.atrous_conv2d( bn4b10_branch2a, w_conv2_20, rate = 2, padding = "SAME", name = "res4b10_branch2b")
                    bn4b10_branch2b = tf.contrib.layers.batch_norm(res4b10_branch2b, is_training = is_training)
                    bn4b10_branch2b = tf.nn.relu( bn4b10_branch2b, name = "bn4b10_branch2b")

                    w_conv3_20 = weight_variable( "W_conv3_20", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b10_branch2c = tf.nn.conv2d( bn4b10_branch2b, w_conv3_20, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b10_branch2c")
                    bn4b10_branch2c = tf.contrib.layers.batch_norm(res4b10_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_20)
                    self._weights.append( w_conv2_20)
                    self._weights.append( w_conv3_20)

                with tf.name_scope( "layer_21" ):
                    res4b10 = tf.add(res4b9_relu, tf.image.resize_bilinear(bn4b10_branch2c, [res4b9_relu.get_shape()[1].value, res4b9_relu.get_shape()[2].value,]), name = "res4b10")
                    res4b10_relu = tf.nn.relu( res4b10, name = "res4b10_relu")

                    w_conv1_21 = weight_variable( "W_conv1_21", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b11_branch2a = tf.nn.conv2d( res4b10_relu, w_conv1_21, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b11_branch2a")
                    bn4b11_branch2a = tf.contrib.layers.batch_norm(res4b11_branch2a, is_training = is_training)
                    bn4b11_branch2a = tf.nn.relu( bn4b11_branch2a, name = "bn4b11_branch2a")

                    w_conv2_21 = weight_variable( "W_conv2_21", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b11_branch2b = tf.nn.atrous_conv2d( bn4b11_branch2a, w_conv2_21, rate = 2, padding = "SAME", name = "res4b11_branch2b")
                    bn4b11_branch2b = tf.contrib.layers.batch_norm(res4b11_branch2b, is_training = is_training)
                    bn4b11_branch2b = tf.nn.relu( bn4b11_branch2b, name = "bn4b11_branch2b")

                    w_conv3_21 = weight_variable( "W_conv3_21", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b11_branch2c = tf.nn.conv2d( bn4b11_branch2b, w_conv3_21, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b11_branch2c")
                    bn4b11_branch2c = tf.contrib.layers.batch_norm(res4b11_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_21)
                    self._weights.append( w_conv2_21)
                    self._weights.append( w_conv3_21)

                with tf.name_scope( "layer_22" ):
                    res4b11 = tf.add(res4b10_relu, tf.image.resize_bilinear(bn4b11_branch2c, [res4b10_relu.get_shape()[1].value, res4b10_relu.get_shape()[2].value,]), name = "res4b11")
                    res4b11_relu = tf.nn.relu( res4b11, name = "res4b11_relu")

                    w_conv1_22 = weight_variable( "W_conv1_22", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b12_branch2a = tf.nn.conv2d( res4b11_relu, w_conv1_22, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b12_branch2a")
                    bn4b12_branch2a = tf.contrib.layers.batch_norm(res4b12_branch2a, is_training = is_training)
                    bn4b12_branch2a = tf.nn.relu( bn4b12_branch2a, name = "bn4b12_branch2a")

                    w_conv2_22 = weight_variable( "W_conv2_22", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b12_branch2b = tf.nn.atrous_conv2d( bn4b12_branch2a, w_conv2_22, rate = 2, padding = "SAME", name = "res4b12_branch2b")
                    bn4b12_branch2b = tf.contrib.layers.batch_norm(res4b12_branch2b, is_training = is_training)
                    bn4b12_branch2b = tf.nn.relu( bn4b12_branch2b, name = "bn4b12_branch2b")

                    w_conv3_22 = weight_variable( "W_conv3_22", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b12_branch2c = tf.nn.conv2d( bn4b12_branch2b, w_conv3_22, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b12_branch2c")
                    bn4b12_branch2c = tf.contrib.layers.batch_norm(res4b12_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_22)
                    self._weights.append( w_conv2_22)
                    self._weights.append( w_conv3_22)

                with tf.name_scope( "layer_23" ):
                    res4b12 = tf.add(res4b11_relu, tf.image.resize_bilinear(bn4b12_branch2c, [res4b11_relu.get_shape()[1].value, res4b11_relu.get_shape()[2].value,]), name = "res4b12")
                    res4b12_relu = tf.nn.relu( res4b12, name = "res4b12_relu")

                    w_conv1_23 = weight_variable( "W_conv1_23", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b13_branch2a = tf.nn.conv2d( res4b12_relu, w_conv1_23, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b13_branch2a")
                    bn4b13_branch2a = tf.contrib.layers.batch_norm(res4b13_branch2a, is_training = is_training)
                    bn4b13_branch2a = tf.nn.relu( bn4b13_branch2a, name = "bn4b13_branch2a")

                    w_conv2_23 = weight_variable( "W_conv2_23", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b13_branch2b = tf.nn.atrous_conv2d( bn4b13_branch2a, w_conv2_23, rate = 2, padding = "SAME", name = "res4b13_branch2b")
                    bn4b13_branch2b = tf.contrib.layers.batch_norm(res4b13_branch2b, is_training = is_training)
                    bn4b13_branch2b = tf.nn.relu( bn4b13_branch2b, name = "bn4b13_branch2b")

                    w_conv3_23 = weight_variable( "W_conv3_23", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b13_branch2c = tf.nn.conv2d( bn4b13_branch2b, w_conv3_23, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b13_branch2c")
                    bn4b13_branch2c = tf.contrib.layers.batch_norm(res4b13_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_23)
                    self._weights.append( w_conv2_23)
                    self._weights.append( w_conv3_23)

                with tf.name_scope( "layer_24" ):
                    res4b13 = tf.add(res4b12_relu, tf.image.resize_bilinear(bn4b13_branch2c, [res4b12_relu.get_shape()[1].value, res4b12_relu.get_shape()[2].value,]), name = "res4b13")
                    res4b13_relu = tf.nn.relu( res4b13, name = "res4b13_relu")

                    w_conv1_24 = weight_variable( "W_conv1_24", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b14_branch2a = tf.nn.conv2d( res4b13_relu, w_conv1_24, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b14_branch2a")
                    bn4b14_branch2a = tf.contrib.layers.batch_norm(res4b14_branch2a, is_training = is_training)
                    bn4b14_branch2a = tf.nn.relu( bn4b14_branch2a, name = "bn4b14_branch2a")

                    w_conv2_24 = weight_variable( "W_conv2_24", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b14_branch2b = tf.nn.atrous_conv2d( bn4b14_branch2a, w_conv2_24, rate = 2, padding = "SAME", name = "res4b14_branch2b")
                    bn4b14_branch2b = tf.contrib.layers.batch_norm(res4b14_branch2b, is_training = is_training)
                    bn4b14_branch2b = tf.nn.relu( bn4b14_branch2b, name = "bn4b14_branch2b")

                    w_conv3_24 = weight_variable( "W_conv3_24", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b14_branch2c = tf.nn.conv2d( bn4b14_branch2b, w_conv3_24, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b14_branch2c")
                    bn4b14_branch2c = tf.contrib.layers.batch_norm(res4b14_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_24)
                    self._weights.append( w_conv2_24)
                    self._weights.append( w_conv3_24)

                with tf.name_scope( "layer_25" ):
                    res4b14 = tf.add(res4b13_relu, tf.image.resize_bilinear(bn4b14_branch2c, [res4b13_relu.get_shape()[1].value, res4b13_relu.get_shape()[2].value,]), name = "res4b14")
                    res4b14_relu = tf.nn.relu( res4b14, name = "res4b14_relu")

                    w_conv1_25 = weight_variable( "W_conv1_25", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b15_branch2a = tf.nn.conv2d( res4b14_relu, w_conv1_25, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b15_branch2a")
                    bn4b15_branch2a = tf.contrib.layers.batch_norm(res4b15_branch2a, is_training = is_training)
                    bn4b15_branch2a = tf.nn.relu( bn4b15_branch2a, name = "bn4b15_branch2a")

                    w_conv2_25 = weight_variable( "W_conv2_25", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b15_branch2b = tf.nn.atrous_conv2d( bn4b15_branch2a, w_conv2_25, rate = 2, padding = "SAME", name = "res4b15_branch2b")
                    bn4b15_branch2b = tf.contrib.layers.batch_norm(res4b15_branch2b, is_training = is_training)
                    bn4b15_branch2b = tf.nn.relu( bn4b15_branch2b, name = "bn4b15_branch2b")

                    w_conv3_25 = weight_variable( "W_conv3_25", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b15_branch2c = tf.nn.conv2d( bn4b15_branch2b, w_conv3_25, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b15_branch2c")
                    bn4b15_branch2c = tf.contrib.layers.batch_norm(res4b15_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_25)
                    self._weights.append( w_conv2_25)
                    self._weights.append( w_conv3_25)

                with tf.name_scope( "layer_26" ):
                    res4b15 = tf.add(res4b14_relu, tf.image.resize_bilinear(bn4b15_branch2c, [res4b14_relu.get_shape()[1].value, res4b14_relu.get_shape()[2].value,]), name = "res4b15")
                    res4b15_relu = tf.nn.relu( res4b15, name = "res4b15_relu")

                    w_conv1_26 = weight_variable( "W_conv1_26", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b16_branch2a = tf.nn.conv2d( res4b15_relu, w_conv1_26, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b16_branch2a")
                    bn4b16_branch2a = tf.contrib.layers.batch_norm(res4b16_branch2a, is_training = is_training)
                    bn4b16_branch2a = tf.nn.relu( bn4b16_branch2a, name = "bn4b16_branch2a")

                    w_conv2_26 = weight_variable( "W_conv2_26", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b16_branch2b = tf.nn.atrous_conv2d( bn4b16_branch2a, w_conv2_26, rate = 2, padding = "SAME", name = "res4b16_branch2b")
                    bn4b16_branch2b = tf.contrib.layers.batch_norm(res4b16_branch2b, is_training = is_training)
                    bn4b16_branch2b = tf.nn.relu( bn4b16_branch2b, name = "bn4b16_branch2b")

                    w_conv3_26 = weight_variable( "W_conv3_26", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b16_branch2c = tf.nn.conv2d( bn4b16_branch2b, w_conv3_26, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b16_branch2c")
                    bn4b16_branch2c = tf.contrib.layers.batch_norm(res4b16_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_26)
                    self._weights.append( w_conv2_26)
                    self._weights.append( w_conv3_26)

                with tf.name_scope( "layer_27" ):
                    res4b16 = tf.add(res4b15_relu, tf.image.resize_bilinear(bn4b16_branch2c, [res4b15_relu.get_shape()[1].value, res4b15_relu.get_shape()[2].value,]), name = "res4b16")
                    res4b16_relu = tf.nn.relu( res4b16, name = "res4b16_relu")

                    w_conv1_27 = weight_variable( "W_conv1_27", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b17_branch2a = tf.nn.conv2d( res4b16_relu, w_conv1_27, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b17_branch2a")
                    bn4b17_branch2a = tf.contrib.layers.batch_norm(res4b17_branch2a, is_training = is_training)
                    bn4b17_branch2a = tf.nn.relu( bn4b17_branch2a, name = "bn4b17_branch2a")

                    w_conv2_27 = weight_variable( "W_conv2_27", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b17_branch2b = tf.nn.atrous_conv2d( bn4b17_branch2a, w_conv2_27, rate = 2, padding = "SAME", name = "res4b17_branch2b")
                    bn4b17_branch2b = tf.contrib.layers.batch_norm(res4b17_branch2b, is_training = is_training)
                    bn4b17_branch2b = tf.nn.relu( bn4b17_branch2b, name = "bn4b17_branch2b")

                    w_conv3_27 = weight_variable( "W_conv3_27", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b17_branch2c = tf.nn.conv2d( bn4b17_branch2b, w_conv3_27, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b17_branch2c")
                    bn4b17_branch2c = tf.contrib.layers.batch_norm(res4b17_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_27)
                    self._weights.append( w_conv2_27)
                    self._weights.append( w_conv3_27)

                with tf.name_scope( "layer_28" ):
                    res4b17 = tf.add(res4b16_relu, tf.image.resize_bilinear(bn4b17_branch2c, [res4b16_relu.get_shape()[1].value, res4b16_relu.get_shape()[2].value,]), name = "res4b17")
                    res4b17_relu = tf.nn.relu( res4b17, name = "res4b17_relu")

                    w_conv1_28 = weight_variable( "W_conv1_28", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b18_branch2a = tf.nn.conv2d( res4b17_relu, w_conv1_28, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b18_branch2a")
                    bn4b18_branch2a = tf.contrib.layers.batch_norm(res4b18_branch2a, is_training = is_training)
                    bn4b18_branch2a = tf.nn.relu( bn4b18_branch2a, name = "bn4b18_branch2a")

                    w_conv2_28 = weight_variable( "W_conv2_28", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b18_branch2b = tf.nn.atrous_conv2d( bn4b18_branch2a, w_conv2_28, rate = 2, padding = "SAME", name = "res4b18_branch2b")
                    bn4b18_branch2b = tf.contrib.layers.batch_norm(res4b18_branch2b, is_training = is_training)
                    bn4b18_branch2b = tf.nn.relu( bn4b18_branch2b, name = "bn4b18_branch2b")

                    w_conv3_28 = weight_variable( "W_conv3_28", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b18_branch2c = tf.nn.conv2d( bn4b18_branch2b, w_conv3_28, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b18_branch2c")
                    bn4b18_branch2c = tf.contrib.layers.batch_norm(res4b18_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_28)
                    self._weights.append( w_conv2_28)
                    self._weights.append( w_conv3_28)

                with tf.name_scope( "layer_29" ):
                    res4b18 = tf.add(res4b17_relu, tf.image.resize_bilinear(bn4b18_branch2c, [res4b17_relu.get_shape()[1].value, res4b17_relu.get_shape()[2].value,]), name = "res4b18")
                    res4b18_relu = tf.nn.relu( res4b18, name = "res4b18_relu")

                    w_conv1_29 = weight_variable( "W_conv1_29", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b19_branch2a = tf.nn.conv2d( res4b18_relu, w_conv1_29, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b19_branch2a")
                    bn4b19_branch2a = tf.contrib.layers.batch_norm(res4b19_branch2a, is_training = is_training)
                    bn4b19_branch2a = tf.nn.relu( bn4b19_branch2a, name = "bn4b19_branch2a")

                    w_conv2_29 = weight_variable( "W_conv2_29", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b19_branch2b = tf.nn.atrous_conv2d( bn4b19_branch2a, w_conv2_29, rate = 2, padding = "SAME", name = "res4b19_branch2b")
                    bn4b19_branch2b = tf.contrib.layers.batch_norm(res4b19_branch2b, is_training = is_training)
                    bn4b19_branch2b = tf.nn.relu( bn4b19_branch2b, name = "bn4b19_branch2b")

                    w_conv3_29 = weight_variable( "W_conv3_29", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b19_branch2c = tf.nn.conv2d( bn4b19_branch2b, w_conv3_29, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b19_branch2c")
                    bn4b19_branch2c = tf.contrib.layers.batch_norm(res4b19_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_29)
                    self._weights.append( w_conv2_29)
                    self._weights.append( w_conv3_29)

                with tf.name_scope( "layer_30" ):
                    res4b19 = tf.add(res4b18_relu, tf.image.resize_bilinear(bn4b19_branch2c, [res4b18_relu.get_shape()[1].value, res4b18_relu.get_shape()[2].value,]), name = "res4b19")
                    res4b19_relu = tf.nn.relu( res4b19, name = "res4b19_relu")

                    w_conv1_30 = weight_variable( "W_conv1_30", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b20_branch2a = tf.nn.conv2d( res4b19_relu, w_conv1_30, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b20_branch2a")
                    bn4b20_branch2a = tf.contrib.layers.batch_norm(res4b20_branch2a, is_training = is_training)
                    bn4b20_branch2a = tf.nn.relu( bn4b20_branch2a, name = "bn4b20_branch2a")

                    w_conv2_30 = weight_variable( "W_conv2_30", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b20_branch2b = tf.nn.atrous_conv2d( bn4b20_branch2a, w_conv2_30, rate = 2, padding = "SAME", name = "res4b20_branch2b")
                    bn4b20_branch2b = tf.contrib.layers.batch_norm(res4b20_branch2b, is_training = is_training)
                    bn4b20_branch2b = tf.nn.relu( bn4b20_branch2b, name = "bn4b20_branch2b")

                    w_conv3_30 = weight_variable( "W_conv3_30", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b20_branch2c = tf.nn.conv2d( bn4b20_branch2b, w_conv3_30, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b20_branch2c")
                    bn4b20_branch2c = tf.contrib.layers.batch_norm(res4b20_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_30)
                    self._weights.append( w_conv2_30)
                    self._weights.append( w_conv3_30)

                with tf.name_scope( "layer_31" ):
                    res4b20 = tf.add(res4b19_relu, tf.image.resize_bilinear(bn4b20_branch2c, [res4b19_relu.get_shape()[1].value, res4b19_relu.get_shape()[2].value,]), name = "res4b20")
                    res4b20_relu = tf.nn.relu( res4b20, name = "res4b20_relu")

                    w_conv1_31 = weight_variable( "W_conv1_31", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b21_branch2a = tf.nn.conv2d( res4b20_relu, w_conv1_31, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b21_branch2a")
                    bn4b21_branch2a = tf.contrib.layers.batch_norm(res4b21_branch2a, is_training = is_training)
                    bn4b21_branch2a = tf.nn.relu( bn4b21_branch2a, name = "bn4b21_branch2a")

                    w_conv2_31 = weight_variable( "W_conv2_31", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b21_branch2b = tf.nn.atrous_conv2d( bn4b21_branch2a, w_conv2_31, rate = 2, padding = "SAME", name = "res4b21_branch2b")
                    bn4b21_branch2b = tf.contrib.layers.batch_norm(res4b21_branch2b, is_training = is_training)
                    bn4b21_branch2b = tf.nn.relu( bn4b21_branch2b, name = "bn4b21_branch2b")

                    w_conv3_31 = weight_variable( "W_conv3_31", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b21_branch2c = tf.nn.conv2d( bn4b21_branch2b, w_conv3_31, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b21_branch2c")
                    bn4b21_branch2c = tf.contrib.layers.batch_norm(res4b21_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_31)
                    self._weights.append( w_conv2_31)
                    self._weights.append( w_conv3_31)

                with tf.name_scope( "layer_32" ):
                    res4b21 = tf.add(res4b20_relu, tf.image.resize_bilinear(bn4b21_branch2c, [res4b20_relu.get_shape()[1].value, res4b20_relu.get_shape()[2].value,]), name = "res4b21")
                    res4b21_relu = tf.nn.relu( res4b21, name = "res4b21_relu")

                    w_conv1_32 = weight_variable( "W_conv1_32", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b22_branch2a = tf.nn.conv2d( res4b21_relu, w_conv1_32, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b22_branch2a")
                    bn4b22_branch2a = tf.contrib.layers.batch_norm(res4b22_branch2a, is_training = is_training)
                    bn4b22_branch2a = tf.nn.relu( bn4b22_branch2a, name = "bn4b22_branch2a")

                    w_conv2_32 = weight_variable( "W_conv2_32", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b22_branch2b = tf.nn.atrous_conv2d( bn4b22_branch2a, w_conv2_32, rate = 2, padding = "SAME", name = "res4b22_branch2b")
                    bn4b22_branch2b = tf.contrib.layers.batch_norm(res4b22_branch2b, is_training = is_training)
                    bn4b22_branch2b = tf.nn.relu( bn4b22_branch2b, name = "bn4b22_branch2b")

                    w_conv3_32 = weight_variable( "W_conv3_32", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b22_branch2c = tf.nn.conv2d( bn4b22_branch2b, w_conv3_32, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b22_branch2c")
                    bn4b22_branch2c = tf.contrib.layers.batch_norm(res4b22_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_32)
                    self._weights.append( w_conv2_32)
                    self._weights.append( w_conv3_32)

                with tf.name_scope( "layer_33" ):
                    res4b22 = tf.add(res4b21_relu, tf.image.resize_bilinear(bn4b22_branch2c, [res4b21_relu.get_shape()[1].value, res4b21_relu.get_shape()[2].value,]), name = "res4b22")
                    res4b22_relu = tf.nn.relu( res4b22, name = "res4b22_relu")

                    w_conv1_33 = weight_variable( "W_conv1_33", shape = [ 1, 1, 1024, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res5a_branch1 = tf.nn.conv2d( res4b22_relu, w_conv1_33, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5a_branch1")
                    bn5a_branch1 = tf.contrib.layers.batch_norm(res5a_branch1, is_training = is_training)

                    self._weights.append( w_conv1_33)

                with tf.name_scope( "layer_34" ):
                    w_conv1_34 = weight_variable( "W_conv1_34", shape = [ 1, 1, 1024, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res5a_branch2a = tf.nn.conv2d( res4b22_relu, w_conv1_34, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5a_branch2a")
                    bn5a_branch2a = tf.contrib.layers.batch_norm(res5a_branch2a, is_training = is_training)
                    bn5a_branch2a = tf.nn.relu( bn5a_branch2a, name = "bn5a_branch2a")

                    w_conv2_34 = weight_variable( "W_conv2_34", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)))
                    res5a_branch2b = tf.nn.atrous_conv2d( bn5a_branch2a, w_conv2_34, rate = 4, padding = "SAME", name = "res5a_branch2b")
                    bn5a_branch2b = tf.contrib.layers.batch_norm(res5a_branch2b, is_training = is_training)
                    bn5a_branch2b = tf.nn.relu( bn5a_branch2b, name = "bn5a_branch2b")

                    w_conv3_34 = weight_variable( "W_conv3_34", shape = [ 1, 1, 512, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res5a_branch2c = tf.nn.conv2d( bn5a_branch2b, w_conv3_34, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5a_branch2c")
                    bn5a_branch2c = tf.contrib.layers.batch_norm(res5a_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_34)
                    self._weights.append( w_conv2_34)
                    self._weights.append( w_conv3_34)

                with tf.name_scope( "layer_35" ):
                    res5a = tf.add(bn5a_branch1, tf.image.resize_bilinear(bn5a_branch2c, [bn5a_branch1.get_shape()[1].value, bn5a_branch1.get_shape()[2].value,]), name = "res5a")
                    res5a_relu = tf.nn.relu( res5a, name = "res5a_relu")

                    w_conv1_35 = weight_variable( "W_conv1_35", shape = [ 1, 1, 2048, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 2048)))
                    res5b_branch2a = tf.nn.conv2d( res5a_relu, w_conv1_35, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5b_branch2a")
                    bn5b_branch2a = tf.contrib.layers.batch_norm(res5b_branch2a, is_training = is_training)
                    bn5b_branch2a = tf.nn.relu( bn5b_branch2a, name = "bn5b_branch2a")

                    w_conv2_35 = weight_variable( "W_conv2_35", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)))
                    res5b_branch2b = tf.nn.atrous_conv2d( bn5b_branch2a, w_conv2_35, rate = 4, padding = "SAME", name = "res5b_branch2b")
                    bn5b_branch2b = tf.contrib.layers.batch_norm(res5b_branch2b, is_training = is_training)
                    bn5b_branch2b = tf.nn.relu( bn5b_branch2b, name = "bn5b_branch2b")

                    w_conv3_35 = weight_variable( "W_conv3_35", shape = [ 1, 1, 512, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res5b_branch2c = tf.nn.conv2d( bn5b_branch2b, w_conv3_35, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5b_branch2c")
                    bn5b_branch2c = tf.contrib.layers.batch_norm(res5b_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_35)
                    self._weights.append( w_conv2_35)
                    self._weights.append( w_conv3_35)

                with tf.name_scope( "layer_36" ):
                    res5b = tf.add(res5a_relu, tf.image.resize_bilinear(bn5b_branch2c, [res5a_relu.get_shape()[1].value, res5a_relu.get_shape()[2].value,]), name = "res5b")
                    res5b_relu = tf.nn.relu( res5b, name = "res5b_relu")

                    w_conv1_36 = weight_variable( "W_conv1_36", shape = [ 1, 1, 2048, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 2048)))
                    res5c_branch2a = tf.nn.conv2d( res5b_relu, w_conv1_36, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5c_branch2a")
                    bn5c_branch2a = tf.contrib.layers.batch_norm(res5c_branch2a, is_training = is_training)
                    bn5c_branch2a = tf.nn.relu( bn5c_branch2a, name = "bn5c_branch2a")

                    w_conv2_36 = weight_variable( "W_conv2_36", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)))
                    res5c_branch2b = tf.nn.atrous_conv2d( bn5c_branch2a, w_conv2_36, rate = 4, padding = "SAME", name = "res5c_branch2b")
                    bn5c_branch2b = tf.contrib.layers.batch_norm(res5c_branch2b, is_training = is_training)
                    bn5c_branch2b = tf.nn.relu( bn5c_branch2b, name = "bn5c_branch2b")

                    w_conv3_36 = weight_variable( "W_conv3_36", shape = [ 1, 1, 512, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res5c_branch2c = tf.nn.conv2d( bn5c_branch2b, w_conv3_36, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5c_branch2c")
                    bn5c_branch2c = tf.contrib.layers.batch_norm(res5c_branch2c, is_training = is_training)

                    self._weights.append( w_conv1_36)
                    self._weights.append( w_conv2_36)
                    self._weights.append( w_conv3_36)

                with tf.name_scope( "layer_37" ):
                    res5c = tf.add(res5b_relu, tf.image.resize_bilinear(bn5c_branch2c, [res5b_relu.get_shape()[1].value, res5b_relu.get_shape()[2].value,]), name = "res5c")
                    res5c_relu = tf.nn.relu( res5c, name = "res5c_relu")

                    w_conv1_37 = weight_variable( "W_conv1_37", shape = [ 3, 3, 2048, self._num_class], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 2048)))
                    fc1_voc12_c0 = tf.nn.atrous_conv2d( res5c_relu, w_conv1_37, rate = 6, padding = "SAME", name = "fc1_voc12_c0")

                    self._weights.append( w_conv1_37)

                with tf.name_scope( "layer_38" ):
                    w_conv1_38 = weight_variable( "W_conv1_38", shape = [ 3, 3, 2048, self._num_class], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 2048)))
                    fc1_voc12_c1 = tf.nn.atrous_conv2d( res5c_relu, w_conv1_38, rate = 12, padding = "SAME", name = "fc1_voc12_c1")

                    self._weights.append( w_conv1_38)

                with tf.name_scope( "layer_39" ):
                    w_conv1_39 = weight_variable( "W_conv1_39", shape = [ 3, 3, 2048, self._num_class], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 2048)))
                    fc1_voc12_c2 = tf.nn.atrous_conv2d( res5c_relu, w_conv1_39, rate = 18, padding = "SAME", name = "fc1_voc12_c2")

                    self._weights.append( w_conv1_39)

                with tf.name_scope( "layer_40" ):
                    w_conv1_40 = weight_variable( "W_conv1_40", shape = [ 3, 3, 2048, self._num_class], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 2048)))
                    fc1_voc12_c3 = tf.nn.atrous_conv2d( res5c_relu, w_conv1_40, rate = 24, padding = "SAME", name = "fc1_voc12_c3")

                    self._weights.append( w_conv1_40)

                with tf.name_scope( "layer_final" ):
                    fc1_voc12 = fc1_voc12_c0 + fc1_voc12_c1 + fc1_voc12_c2 + fc1_voc12_c3
                    fc1_voc12 = tf.image.resize_bilinear(fc1_voc12, [self._output_HW[0], self._output_HW[1]])


        self._logits = fc1_voc12
        
        self._predictor = self.pixel_wise_softmax_2( self._logits)
        #self._predictor = tf.nn.log_softmax( self._logits)
        self._saver = tf.train.Saver( max_to_keep = None)


    def pixel_wise_softmax_2( self, output_map):
        tensor_max = tf.tile( tf.reduce_max( output_map, 3, keep_dims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        exponential_map = tf.exp( output_map - tensor_max)
        tensor_sum_exp = tf.tile( tf.reduce_sum( exponential_map, 3, keep_dims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        """
        logits = tf.div(exponential_map,tensor_sum_exp)
        square_map = tf.square(logits)
        sum_square = tf.reduce_sum(square_map,3, keep_dims = True)
        tensor_sum_square = tf.tile(sum_square, tf.stack([1,1,1,tf.shape(logits)[3]]))
        return tf.div(square_map,tensor_sum_square)
        """
        return tf.div( exponential_map, tensor_sum_exp, name = "predictor")

    def add_L2_regularizer( self, cost):
        
        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum( [ tf.nn.l2_loss( variable) for variable in self._weights + self._biases])
            cost += (regularizer * regularizers)
            
        return cost


    def train( self, data, output_path, training_iters = 10, epochs = 100, keep_prob = 0.75, display_step = 1, opt_kwargs = {}):
        
        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_train_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        use_weight_map = opt_kwargs.pop( "use_weight_map", False)
        cost_name = opt_kwargs.pop("cost", "dice_coefficient")
        optimizer_name = opt_kwargs.pop( "optimizer", "SGD")
        learning_rate = opt_kwargs.pop( "learning_rate", 0.2)
        decay_rate = opt_kwargs.pop( "decay_rate", 0.95)
        momentum = opt_kwargs.pop( "momentum", 0.2)
        batch_size = opt_kwargs.pop( "batch_size", 1)
        verification_path = opt_kwargs.pop( "verification_path", "verification")
        verification_batch_size = opt_kwargs.pop( "verification_batch_size", 4)
        pre_trained_model_iteration = opt_kwargs.pop( "pre_trained_model_iteration", None)
        test_data = opt_kwargs.pop( "test_data", None)
        use_average_mirror = opt_kwargs.pop( "use_average_mirror", False)
        save_model_epochs = opt_kwargs.pop( "save_model_epochs", np.arange( epochs))
        func_save_conditonal_model = opt_kwargs.pop( "func_save_conditonal_model", None)
        additional_str = opt_kwargs.pop( "additional_str", None)

        experimental_pmap_threshold_val = opt_kwargs.pop( "experimental_pmap_threshold_val", -1)
        """

        # get options -----
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_train_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        use_weight_map = opt_kwargs.pop( "use_weight_map", False)
        cost_name = opt_kwargs.pop("cost", "dice_coefficient")
        optimizer_name = opt_kwargs.pop( "optimizer", "SGD")
        learning_rate = opt_kwargs.pop( "learning_rate", 0.2)
        decay_rate = opt_kwargs.pop( "decay_rate", 0.95)
        momentum = opt_kwargs.pop( "momentum", 0.2)
        batch_size = opt_kwargs.pop( "batch_size", 1)
        verification_path = opt_kwargs.pop( "verification_path", "verification")
        verification_batch_size = opt_kwargs.pop( "verification_batch_size", 4)
        pre_trained_model_iteration = opt_kwargs.pop( "pre_trained_model_iteration", None)
        test_data = opt_kwargs.pop( "test_data", None)
        use_average_mirror = opt_kwargs.pop( "use_average_mirror", False)
        save_model_epochs = opt_kwargs.pop( "save_model_epochs", np.arange( epochs))
        func_save_conditonal_model = opt_kwargs.pop( "func_save_conditonal_model", None)
        additional_str = opt_kwargs.pop( "additional_str", None)
        
        experimental_pmap_threshold_val = opt_kwargs.pop( "experimental_pmap_threshold_val", -1)
        # get options =====


        if len( opt_kwargs):
            raise ValueError( "wrong opt_kwargs : %s" % ( str( opt_kwargs.keys())))

        if len( logger.handlers) == 2:
            logger.removeHandler( logger.handlers[ -1])

        if os.path.isdir( logging_folder) == False:
            os.makedirs( logging_folder)
        file_handler = logging.FileHandler( os.path.join( logging_folder, logging_name))
        file_handler.setFormatter( log_formatter)
        logger.addHandler( file_handler)

        logger_batch_loss = logging.getLogger( "train_batch_loss")
        logger_batch_loss.setLevel( logging.INFO)
        file_handler_bl = logging.FileHandler( os.path.join( logging_folder, os.path.splitext( logging_name)[ 0] + "_loss.log"))
        file_handler_bl.setFormatter( log_formatter)
        logger_batch_loss.addHandler( file_handler_bl)

        logging_str = [ "training_data_params >>\n",
                       data.get_log()]
        logger.info( ''.join( logging_str))

        logging_str = [ "test_data_params >>\n",
                       test_data.get_log() if test_data is not None else "\t\tNone"]
        logger.info( ''.join( logging_str))

        logging_str = [ "train_parmas >>\n",
                        "\t\t\tlogging name : {0}\n".format( logging_name),
                        "\t\t\tnum_channel : {0}\n".format( self._num_channel),
                        "\t\t\tnum_class : {0}\n".format( self._num_class),
                        "\t\t\toutput_HW : {0}\n".format( self._output_HW),
                        "\t\t\toutput_path : {0}\n".format( output_path),
                        "\t\t\ttraining_iters : {0}\n".format( training_iters),
                        "\t\t\tepochs : {0}\n".format( epochs),
                        "\t\t\tkeep_prob : {0}\n".format( keep_prob),
                        "\t\t\tdisplay_step : {0}\n".format( display_step),
                        "\t\t\tcost_name : {0}\n".format( cost_name),
                        "\t\t\toptimizer_name : {0}\n".format( optimizer_name),
                        "\t\t\tlearning_rate : {0}\n".format( learning_rate),
                        "\t\t\tdecay_rate : {0}\n".format( decay_rate) if optimizer_name == "SGD" else "",
                        "\t\t\tmomentum : {0}\n".format( momentum) if optimizer_name == "SGD" else "",
                        "\t\t\tbatch_size : {0}\n".format( batch_size),
                        "\t\t\tverification_path : {0}\n".format( verification_path),
                        "\t\t\tverification_batch_size : {0}\n".format( verification_batch_size),
                        "\t\t\tpre_trained_model_iteration : {0}\n".format( str( pre_trained_model_iteration) if pre_trained_model_iteration is not None else "None"),
                        "\t\t\tsave_model_epochs : {0}\n".format( save_model_epochs),
                        "\t\t\taddtional_str : {0}\n".format( additional_str) if additional_str is not None else ""]
        logger.info( ''.join( logging_str))


        logging_str = [ "train_parmas experimental >>\n",
                        "\t\t\texperimental_pmap_threshold_val : {0}\n".format( experimental_pmap_threshold_val)]
        logger.info( ''.join( logging_str))


        for weight in self._weights:
            sidx = find_nth( weight.name, '/', 2) + 1
            tf.summary.histogram( name = weight.name[ sidx : -2], values = weight)
        for bias in self._biases:
            sidx = find_nth( bias.name, '/', 2) + 1
            tf.summary.histogram( name = bias.name[ sidx : -2], values = bias)

        shutil.rmtree( verification_path, ignore_errors = True)
        time.sleep( 0.100)
        os.makedirs( verification_path, exist_ok = True)
        
        if experimental_pmap_threshold_val > 0:
            prediction0 = self.pixel_wise_softmax_2( self._logits)
            logits_comparison = tf.less( prediction0, tf.constant( experimental_pmap_threshold_val, dtype = tf.float32))
            prediction = tf.where( logits_comparison, tf.zeros_like( prediction0), prediction0)
        else:
            prediction = self.pixel_wise_softmax_2( self._logits)
            #prediction = tf.nn.log_softmax( self._logits)


        if cost_name == "cross_entropy" :
            
            flat_logits = tf.reshape( self._logits, [-1, self._num_class])
            flat_labels = tf.reshape( self._y, [-1, self._num_class])
            if use_weight_map == False:
                cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = flat_logits, labels = flat_labels))
            else:
                flat_loss_map = tf.nn.softmax_cross_entropy_with_logits( logits = flat_logits, labels = flat_labels)
                loss_map = tf.reshape( flat_loss_map, [ -1, self._output_HW[ 0], self._output_HW[ 1]])
                #flat_weight_map = tf.reshape( self._weight_map, [ -1])
                cost = tf.reduce_mean( loss_map * self._weight_map)
        elif cost_name == "dice_coefficient":
            use_weight_map = False
            eps = 1e-5
            for nc in range( self._num_class):    
                prediction_nc = prediction[ :, :, :, nc]
                
                #prediction_nc = tf.cast(prediction_nc+0.5, tf.float32)
                
                intersection = tf.reduce_sum( prediction_nc * self._y[ :, :, :, nc])
                union = eps + tf.reduce_sum( prediction_nc * prediction_nc) + tf.reduce_sum( self._y[ :, :, :, nc])

                if "cost" in locals():
                    cost += -( 2 * intersection / union)
                else:
                    cost = -( 2 * intersection / union)
            cost /= self._num_class

        elif cost_name == "bootstrap":
            xentropy = -tf.reduce_sum(prediction * self._y, axis = 3)
            K = 512 * 64

            result = tf.constant(0, dtype = tf.float32)
            for i in range(batch_size):
                batch_errors = xentropy[i]
                #flat_errors = tf.contrib.layers.flatten(batch_errors)
                flat_errors = tf.reshape( batch_errors, [ -1, 400 * 400])
                worst_errors, indices_top = tf.nn.top_k(flat_errors, k = K, sorted = False)
                result += tf.reduce_mean(worst_errors)
            result /= tf.constant(batch_size, dtype= tf.float32)
            cost = result

        elif cost_name == "mean_square":
            use_weight_map = False
            loss = tf.reduce_sum((prediction - self._y)*(prediction - self._y))
            size = (tf.shape(prediction)[3]*tf.shape(prediction)[1]*tf.shape(prediction)[2]*tf.shape(prediction)[0])
            cost = tf.sqrt(loss/tf.cast(size, tf.float32))

                
        global_step = tf.Variable( 0, name = "global_step", trainable = False)
        if optimizer_name == "SGD":
            learning_rate_node = tf.train.exponential_decay( learning_rate = learning_rate, global_step = global_step, decay_steps = training_iters, decay_rate = decay_rate, staircase = True)
            optimizer = tf.train.MomentumOptimizer( learning_rate = learning_rate_node, momentum = momentum).minimize( cost, global_step = global_step)
        else:
            adam_op = tf.train.AdamOptimizer( learning_rate = learning_rate)
            optimizer = adam_op.minimize( cost, global_step = global_step)
            learning_rate_node = adam_op._lr_t

        
        tf.summary.scalar( "loss", cost)
        tf.summary.scalar( "learning_rate", learning_rate_node)
        self._summary_op = tf.summary.merge_all()
        
        with tf.Session() as sess:
            
            summary_writer = tf.summary.FileWriter( output_path, graph = sess.graph)

            logger.info( "Start optimization")

            sess.run( tf.global_variables_initializer())

            if pre_trained_model_iteration == None:
                start_iter = 0
            else:
                start_iter = self.restore( sess, output_path, pre_trained_model_iteration)

            if -1 in save_model_epochs:
                self.save( sess, output_path, 0.0, start_iter)
                
            pad_shape0 = ( 0, 0)
            pad_shape1 = ( 0, 0)

            verification_window_rect = ( 0, 0, self._output_HW[ 0], self._output_HW[ 1])
            verification_x, verification_y, verification_weight = data.get_window_batch( pad_shape0, pad_shape1, verification_window_rect, 0, verification_batch_size)
            
            verification_pr, error_rate = self.output_verification_stats( sess, cost, use_weight_map, verification_x, verification_y, verification_weight)
            data.save_prediction_img( verification_path, "_init", verification_x, verification_y, verification_pr, mask = None)

            batch_x = np.ndarray( shape = ( ( batch_size,) + self._input_HW + ( self._num_channel,)), dtype = np.float32)
            batch_y = np.ndarray( shape = ( ( batch_size,) + self._output_HW + ( self._num_class,)), dtype = np.float32)
            batch_weight = np.ndarray( shape = ( ( batch_size,) + self._output_HW), dtype = np.float32)    
            batch_fname = []
            for epoch in range( epochs):
                total_loss = 0
                for step in range( ( epoch * training_iters), ( ( epoch + 1) * training_iters)):

                    for nb in range( batch_size):
                        x0, y0, weight0, mask0, fname = data.next( pad_shape0, pad_shape1)
                        batch_x[ nb, :, :, :] = x0
                        batch_y[ nb, :, :, :] = y0
                        batch_weight[ nb, :, :] = weight0
                        #batch_fname.append( ( fname, window_rect))
 
                    if use_weight_map == False:
                        _, loss, lr = sess.run( ( optimizer, cost, learning_rate_node), feed_dict = { self._x: batch_x,
                                                                                                    self._y: batch_y,
                                                                                                    self._keep_prob: keep_prob})
                    else:
                        _, loss, lr = sess.run( ( optimizer, cost, learning_rate_node), feed_dict = { self._x: batch_x,
                                                                                                    self._y: batch_y,
                                                                                                    self._weight_map: batch_weight,
                                                                                                    self._keep_prob: keep_prob})
                    logging_bl_str = []
                    logging_bl_str.append( "loss = %f\n" % ( loss))
                    for info in batch_fname:
                        logging_bl_str.append( "\t\t\tname = {:>12}, x = {:>5}, y = {:>5}, w = {:>5}, h = {:>5}\n".format( info[ 0], info[ 1][ 1], info[ 1][ 0], info[ 1][ 3], info[ 1][ 2]))
                    logger_batch_loss.info( ''.join( logging_bl_str))
                    batch_fname = []

                    
                    if step % display_step == 0:
                        self.output_minibatch_stats( sess, cost, use_weight_map, start_iter + step, batch_x, batch_y, batch_weight)
                        
                    total_loss += loss

                

                if use_weight_map == False:
                    summary_str = sess.run( self._summary_op, feed_dict = { self._x: batch_x, self._y: batch_y, self._keep_prob: 1.})
                else:
                    summary_str = sess.run( self._summary_op, feed_dict = { self._x: batch_x, self._y: batch_y, self._weight_map: batch_weight, self._keep_prob: 1.})
                summary_writer.add_summary( summary_str, epoch)
                summary_writer.flush()

                logger.info( "Epoch {:}, Average loss: {:.4f}, learning rate: {:e}".format( epoch, ( total_loss / training_iters), lr))
                verification_pr, error_rate = self.output_verification_stats( sess, cost, use_weight_map, verification_x, verification_y, verification_weight)
                data.save_prediction_img( verification_path, "epoch_%s" % epoch, verification_x, verification_y, verification_pr, mask = None)
                 
                if test_data is not None:
                    error_rate, cm = self.output_test_stats( sess, cost, use_weight_map, test_data, use_average_mirror)

                if epoch in save_model_epochs:
                    self.save( sess, output_path, error_rate, start_iter + ( epoch) * training_iters)

                if func_save_conditonal_model is not None and test_data is not None:
                    save_paths = func_save_conditonal_model( epoch, cm)
                    
                    for save_path in save_paths:
                        save_conditional_model_path = os.path.join( output_path, save_path)
                        shutil.rmtree( save_conditional_model_path, ignore_errors = True)
                        time.sleep( 0.100)
                        os.makedirs( save_conditional_model_path, exist_ok = True)
                        self.save( sess, save_conditional_model_path, error_rate, start_iter + ( epoch) * training_iters)
            logger.info("Optimization Finished!")
            
            return output_path
        
    
    def test( self, data, output_img_path, model_path, model_iter = -1, opt_kwargs = {}):

        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_test_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        use_average_mirror = opt_kwargs.pop( "use_average_mirror", False)
        step_width = opt_kwargs.pop( "step_width", -1)
        step_height = opt_kwargs.pop( "step_height", -1)
        save_img_type = opt_kwargs.pop( "save_img_type", 0)
        return_gts = opt_kwargs.pop( "return_gts", False)
        return_prs = opt_kwargs.pop( "return_prs", False)
        additional_str = opt_kwargs.pop( "additional_str", None)
        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_test_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        use_average_mirror = opt_kwargs.pop( "use_average_mirror", False)
        step_width = opt_kwargs.pop( "step_width", -1)
        step_height = opt_kwargs.pop( "step_height", -1)
        save_img_type = opt_kwargs.pop( "save_img_type", 0)
        return_gts = opt_kwargs.pop( "return_gts", False)
        return_prs = opt_kwargs.pop( "return_prs", False)
        additional_str = opt_kwargs.pop( "additional_str", None)

        if len( opt_kwargs):
            raise ValueError( "wrong opt_kwargs : %s" % ( str( opt_kwargs.keys())))

        if len( logger.handlers) == 2:
            logger.removeHandler( logger.handlers[ -1])

        if os.path.isdir( logging_folder) == False:
            os.makedirs( logging_folder)
        file_handler = logging.FileHandler( os.path.join( logging_folder, logging_name))
        file_handler.setFormatter( log_formatter)
        logger.addHandler( file_handler)

        logging_str = [ "test_parmas >>\n",
                        "\t\t\tlogging name : {0}\n".format( logging_name),
                        "\t\t\tdata_module : {0}\n".format( data.__module__),
                        "\t\t\tdata_path : {0}\n".format( data.dir),
                        "\t\t\toutput_img_path : {0}\n".format( output_img_path),
                        "\t\t\tmodel_path : {0}\n".format( model_path),
                        "\t\t\tuse_average_mirror : {0}\n".format( use_average_mirror),
                        "\t\t\tmoving_width : {0}\n".format( step_width),
                        "\t\t\tmoving_height : {0}\n".format( step_height),
                        "\t\t\tadditional_str : {0}\n".format( additional_str) if additional_str is not None else ""]
        logger.info( ''.join( logging_str))
        

        shutil.rmtree( output_img_path, ignore_errors = True)
        time.sleep( 0.100)
        os.makedirs( output_img_path, exist_ok = True)
        f_csv = open( os.path.join( output_img_path, "result.csv"), "wt")

        with tf.Session() as sess:
                        
            sess.run( tf.global_variables_initializer())
            self.restore( sess, model_path, model_iter)

            pad_shape0 = ( 0, 0)
            pad_shape1 = ( 0, 0)
            
            if return_gts == True:
                gts = np.zeros( shape = ( data.num_examples, data._resize_shape[ 0], data._resize_shape[ 1], data.num_class), dtype = np.float32)    
            if return_prs == True:
                prs = np.zeros( shape = ( data.num_examples, data._resize_shape[ 0], data._resize_shape[ 1], data.num_class), dtype = np.float32)    

            data_num_class_wo_fake = data.num_class_wo_fake
            total_pixel_error = 0.
            ACCURACY = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
            PRECISION = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
            TP = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
            TN = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
            DS = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
            confusion_matrix_by_class = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
            for nd in range( data.num_examples):

                x0, y0, _, mask0 = data.get( pad_shape0, pad_shape1, nd)

                shape = y0.shape[ 1 : 3]

                x = np.ndarray( shape = ( 1,) + shape + ( data.num_channel,), dtype = np.float32)
                pr = np.zeros( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)
                acc_pr = np.zeros( shape = ( 1,) + shape + ( 1,), dtype = np.float32)

                
                if mask0 is not None:
                    new_window_rects = []
                    for nw, wr in enumerate( window_rects):
                        on_cnt = np.count_nonzero( mask0[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3]])
                        #if on_cnt > np.prod( self._output_HW) * 0.1:
                        if on_cnt > np.prod( self._output_HW) * 0.01:
                            new_window_rects.append( wr)
                    window_rects = new_window_rects


                wx = x0
                wy = y0
                pr_ = sess.run( self._predictor, feed_dict = { self._x: wx, self._y: wy, self._keep_prob: 1.})
                x = wx
                pr += pr_
                acc_pr += 1
                

                if type( use_average_mirror) == bool and use_average_mirror == True:

                    mirrored_x0s = [ np.flip( x0, axis = 1), np.flip( x0, axis = 2), np.flip( np.flip( x0, axis = 2), 1)]
                    mirrored_y0s = [ np.flip( y0, axis = 1), np.flip( y0, axis = 2), np.flip( np.flip( y0, axis = 2), 1)]
                    mirrored_prs = [ np.zeros_like( pr), np.zeros_like( pr), np.zeros_like( pr)]
                    for mirrored_x0, mirrored_y0, mirrored_pr in zip( mirrored_x0s, mirrored_y0s, mirrored_prs):
                        wx = mirrored_x0
                        wy = mirrored_y0
                        pr_ = sess.run( self._predictor, feed_dict = { self._x: wx, self._y: wy, self._keep_prob: 1.})
                        mirrored_pr += pr_
                    
                    pr += np.flip( mirrored_prs[ 0], axis = 1)
                    pr += np.flip( mirrored_prs[ 1], axis = 2)
                    pr += np.flip( np.flip( mirrored_prs[ 2], axis = 1), axis = 2)
                    acc_pr = acc_pr + np.flip( acc_pr, axis = 1) + np.flip( acc_pr, axis = 2) + np.flip( np.flip( acc_pr, axis = 2), 1)
                elif type( use_average_mirror) == list:

                    mirrored_x0s = []
                    mirrored_y0s = []
                    mirrored_prs = []
                    for nf in use_average_mirror:
                        mirrored_x0s.append( np.flip( x0, axis = nf))
                        mirrored_y0s.append( np.flip( y0, axis = nf))
                        mirrored_prs.append( np.zeros_like( pr))
                    for mirrored_x0, mirrored_y0, mirrored_pr in zip( mirrored_x0s, mirrored_y0s, mirrored_prs):
                        for nw, wr in enumerate( window_rects):
                            wx = mirrored_x0[ :, wr[ 0] : wr[ 0] + wr[ 2] + np.sum( pad_shape0), wr[ 1] : wr[ 1] + wr[ 3] + np.sum( pad_shape1), :]
                            wy = mirrored_y0[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :]
                            pr_ = sess.run( self._predictor, feed_dict = { self._x: wx, self._y: wy, self._keep_prob: 1.})
                            mirrored_pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += pr_
                    
                    for ni, nf in enumerate( use_average_mirror):
                        pr += np.flip( mirrored_prs[ ni], axis = nf)
                        acc_pr = acc_pr + np.flip( acc_pr, axis = nf)

                pr = pr / acc_pr
                if return_gts == True:
                    gts[ nd, ...] = y0[ 0, ...]
                if return_prs == True:
                    prs[ nd, ...] = pr[ 0, ...]
                argmax_pr = np.argmax( pr, 3)
                argmax_gt = np.argmax( y0, 3)
                argmax_pr_ncs = []
                argmax_gt_ncs = []
                for nc in range( data.num_class):
                    argmax_pr_ncs.append( argmax_pr == nc)
                    argmax_gt_ncs.append( argmax_gt == nc)

                for nc in range( data_num_class_wo_fake):
                    argmax_pr_nc = argmax_pr_ncs[ nc]
                    argmax_gt_nc = argmax_gt_ncs[ nc]
                    tp = np.count_nonzero( np.logical_and( argmax_pr_nc, argmax_gt_nc))
                    tn = np.count_nonzero( np.logical_and( ( ~argmax_pr_nc), ( ~argmax_gt_nc)))
                    union = np.count_nonzero( np.logical_or( argmax_pr_nc, argmax_gt_nc))
                    tp_fp = np.count_nonzero( argmax_pr_nc)
                    tp_fn = np.count_nonzero( argmax_gt_nc)
                    not_tp_fn = np.count_nonzero( ~argmax_gt_nc)

                    PRECISION[ nd, nc] = ( tp / tp_fp) if tp_fp > 0 else np.nan
                    ACCURACY[ nd, nc] = ( tp / union) if union > 0 else np.nan
                    TP[ nd, nc] = ( tp / tp_fn) if tp_fn > 0 else np.nan
                    TN[ nd, nc] = ( tn / not_tp_fn) if not_tp_fn > 0 else np.nan
                    DS[ nd, nc] = ( 2 * tp / ( tp_fp + tp_fn)) if tp_fp + tp_fn > 0 else np.nan
                
                # confusion-matrix by class
                for nc_gt in range( data.num_class):
                    for nc_pr in range( data.num_class):
                        cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                        confusion_matrix_by_class[ nc_gt][ nc_pr] += cm_val

                pixel_error = 100.0 * np.count_nonzero( argmax_pr != argmax_gt) / ( 1 * pr.shape[ 1] * pr.shape[ 2])
                total_pixel_error += pixel_error

                logging_str = [ "image_name = {:}\n".format( data.img_list[ nd]),
                               "\t\t\tpixel_error = {:.2f}%\n".format( pixel_error),
                                "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY[ nd, :])),
                                "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP[ nd, :])),
                                "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION[ nd, :])),
                                "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN[ nd, :])),
                                "\t\t\tdice_similarity = {:.2f}".format( np.nanmean( DS[ nd, :]))]
                logger.info( ''.join( logging_str))
                f_csv.write( "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % ( data.img_list[ nd], pixel_error, np.nanmean( ACCURACY[ nd, :]), np.nanmean( TP[ nd, :]), np.nanmean( PRECISION[ nd, :]), np.nanmean( TN[ nd, :]), np.nanmean( DS[ nd, :])))

                #x = np.pad( x, ( ( 0, 0), pad_shape0, pad_shape1, ( 0, 0)), mode = "constant")
                result_img_name, _ = os.path.splitext( data.img_list[ nd])
                data.save_prediction_img( output_img_path, result_img_name, x0, y0, pr, save_img_type = save_img_type, mask = mask0)
                
            logging_str = [ "total mean>>\n",
                            "\t\t\tpixel_error = {:.2f}%\n".format( total_pixel_error / data.num_examples),
                            "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY)),
                            "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP)),
                            "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION)),
                            "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN)),
                            "\t\t\tdice_similarity = {:.2f}".format( np.nanmean( DS))]
            logger.info( ''.join( logging_str))
            formatter = '[' + ( "{:6d}," * data.num_class)[ : -1] + ']'
            logging_str = [ "========================================================\n",
                            "\t\t\taccuracy = {0}\n".format( np.array2string( np.nanmean( ACCURACY, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                            "\t\t\trecall = {0}\n".format( np.array2string( np.nanmean( TP, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                            "\t\t\tprecision = {0}\n".format( np.array2string( np.nanmean( PRECISION, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                            "\t\t\ttrue_negatives = {0}\n".format( np.array2string( np.nanmean( TN, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                            "\t\t\tdice_similarity = {0}\n".format( np.array2string( np.nanmean( DS, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                            "\t\t\tconfusion_matrix_nc>>\n",
                            *[ "\t\t\t\t%s\n" % ( formatter.format( *[ confusion_matrix_by_class[ nc1][ nc2] for nc2 in range( data.num_class)])) for nc1 in range( data.num_class)],
                            "\t\t\trecall_by_cm = {0}\n".format( np.array2string( np.array( [ confusion_matrix_by_class[ nc][ nc] / np.sum( confusion_matrix_by_class[ nc, :]) for nc in range( data.num_class)]), max_line_width = 1000, precision = 4, separator = ',')),
                            "\t\t\tprecision_by_cm = {0}\n".format( np.array2string( np.array( [ confusion_matrix_by_class[ nc][ nc] / np.sum( confusion_matrix_by_class[ :, nc]) for nc in range( data.num_class)]), max_line_width = 1000, precision = 4, separator = ','))]
            logger.info( ''.join( logging_str))

            f_csv.write( "\n\n")
            f_csv.write( "mean_nc_accuracy,%s\n" % ( np.array2string( np.nanmean( ACCURACY, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_nc_recall,%s\n" % ( np.array2string( np.nanmean( TP, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_nc_precision,%s\n" % ( np.array2string( np.nanmean( PRECISION, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_nc_dice_similarity,%s\n" % ( np.array2string( np.nanmean( DS, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "\n\n")
            f_csv.write( "mean_accuracy,%s\n" % ( np.array2string( np.nanmean( ACCURACY), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_recall,%s\n" % ( np.array2string( np.nanmean( TP), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_precision,%s\n" % ( np.array2string( np.nanmean( PRECISION), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_true_negatives,%s\n" % ( np.array2string( np.nanmean( TN), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "mean_dice_similarity,%s\n" % ( np.array2string( np.nanmean( DS), max_line_width = 1000, precision = 4, separator = ',')))
            f_csv.write( "\n\nconfusion_matrix_nc\n")
            f_csv.write( "%s\n" % ( np.array2string( confusion_matrix_by_class, max_line_width = 1000, separator = ',')))
        f_csv.close()
        return ( gts if return_gts == True else None, prs if return_prs == True else None)


    def test_from_prs( self, data, gts, arr_prs, output_img_path, opt_kwargs = {}):
        
        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_test_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        ensemble_type = opt_kwargs.pop( "ensemble_type", "ADD")
        save_img_type = opt_kwargs.pop( "save_img_type", 0)
        imgs = opt_kwargs.pop( "imgs", None)
        additional_str = opt_kwargs.pop( "additional_str", None)
        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_test_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        ensemble_type = opt_kwargs.pop( "ensemble_type", "ADD")
        save_img_type = opt_kwargs.pop( "save_img_type", 0)
        imgs = opt_kwargs.pop( "imgs", None)
        additional_str = opt_kwargs.pop( "additional_str", None)

        if len( opt_kwargs):
            raise ValueError( "wrong opt_kwargs : %s" % ( str( opt_kwargs.keys())))

        if len( logger.handlers) == 2:
            logger.removeHandler( logger.handlers[ -1])

        if os.path.isdir( logging_folder) == False:
            os.makedirs( logging_folder)
        file_handler = logging.FileHandler( os.path.join( logging_folder, logging_name))
        file_handler.setFormatter( log_formatter)
        logger.addHandler( file_handler)

        logging_str = [ "test_parmas >>\n",
                        "\t\t\tlogging name : {0}\n".format( logging_name),
                        "\t\t\toutput_img_path : {0}\n".format( output_img_path),
                        "\t\t\tensemble_type : {0}\n".format( ensemble_type),
                        "\t\t\tadditional_str : {0}\n".format( additional_str) if additional_str is not None else ""]
        logger.info( ''.join( logging_str))
        
        prs = np.zeros_like( arr_prs[ 0])
        if ensemble_type == "ADD":
            for prs0 in arr_prs:
                prs += prs0
            prs /= len( arr_prs)
        else:
            """
            prs_or = np.full( shape = arr_prs[ 0].shape, fill_value = False, dtype = np.bool)
            for prs0 in arr_prs:
                prs_or[ ..., 0] = np.logical_or( prs_or[ ..., 0], prs0[ ..., 0] >= 0.5)
            for nc in range( 1, prs_or.shape[ 3]):
                prs_or[ ..., nc] = np.logical_not( prs_or[ ..., 0])
            prs = prs_or.astype( np.float32)
            """
            prs_max = np.full( shape = arr_prs[ 0].shape, fill_value = 0, dtype = np.float32)
            for prs0 in arr_prs:
                prs_max[ ..., 0] = np.max( ( prs_max[ ..., 0], prs0[ ..., 0]), axis = 0)
            for nc in range( 1, prs_max.shape[ 3]):
                prs_max[ ..., nc] = ( 1 - prs_max[ ..., 0]) / ( prs_max.shape[ 3] - 1)
            prs = prs_max
            

        shutil.rmtree( output_img_path, ignore_errors = True)
        time.sleep( 0.100)
        os.makedirs( output_img_path, exist_ok = True)
        f_csv = open( os.path.join( output_img_path, "result.csv"), "wt")
                                
        pad_shape0 = ( ( self._input_HW[ 0] - self._output_HW[ 0]) // 2, ( self._input_HW[ 0] - self._output_HW[ 0]) // 2)
        pad_shape1 = ( ( self._input_HW[ 1] - self._output_HW[ 1]) // 2, ( self._input_HW[ 1] - self._output_HW[ 1]) // 2)
                
        total_pixel_error = 0.
        ACCURACY = np.ndarray( shape = ( data.num_examples, data.num_class_wo_fake), dtype = np.float32)
        PRECISION = np.ndarray( shape = ( data.num_examples, data.num_class_wo_fake), dtype = np.float32)
        TP = np.ndarray( shape = ( data.num_examples, data.num_class_wo_fake), dtype = np.float32)
        TN = np.ndarray( shape = ( data.num_examples, data.num_class_wo_fake), dtype = np.float32)
        DS = np.ndarray( shape = ( data.num_examples, data.num_class_wo_fake), dtype = np.float32)
        confusion_matrix_by_class = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
        for nd in range( data.num_examples):

            pr = prs[ nd]
            pr = pr[ np.newaxis, ...]
            y0 = gts[ nd]
            y0 = y0[ np.newaxis, ...]

            argmax_pr = np.argmax( pr, 3)
            argmax_gt = np.argmax( y0, 3)
            argmax_pr_ncs = []
            argmax_gt_ncs = []
            for nc in range( data.num_class):
                argmax_pr_ncs.append( argmax_pr == nc)
                argmax_gt_ncs.append( argmax_gt == nc)

            for nc in range( data.num_class_wo_fake):
                argmax_pr_nc = argmax_pr_ncs[ nc]
                argmax_gt_nc = argmax_gt_ncs[ nc]
                tp = np.count_nonzero( np.logical_and( argmax_pr_nc, argmax_gt_nc))
                tn = np.count_nonzero( np.logical_and( ( ~argmax_pr_nc), ( ~argmax_gt_nc)))
                union = np.count_nonzero( np.logical_or( argmax_pr_nc, argmax_gt_nc))
                tp_fp = np.count_nonzero( argmax_pr_nc)
                tp_fn = np.count_nonzero( argmax_gt_nc)
                not_tp_fn = np.count_nonzero( ~argmax_gt_nc)

                PRECISION[ nd, nc] = ( tp / tp_fp) if tp_fp > 0 else np.nan
                ACCURACY[ nd, nc] = ( tp / union) if union > 0 else np.nan
                TP[ nd, nc] = ( tp / tp_fn) if tp_fn > 0 else np.nan
                TN[ nd, nc] = ( tn / not_tp_fn) if not_tp_fn > 0 else np.nan
                DS[ nd, nc] = ( 2 * tp / ( tp_fp + tp_fn)) if tp_fp + tp_fn > 0 else np.nan
                
            # confusion-matrix by class
            for nc_gt in range( data.num_class):
                for nc_pr in range( data.num_class):
                    cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                    confusion_matrix_by_class[ nc_gt][ nc_pr] += cm_val

            pixel_error = 100.0 * np.count_nonzero( argmax_pr != argmax_gt) / ( 1 * pr.shape[ 1] * pr.shape[ 2])
            total_pixel_error += pixel_error

            logging_str = [ "image_name = {:}\n".format( data.img_list[ nd]),
                            "\t\t\tpixel_error = {:.2f}%\n".format( pixel_error),
                            "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY[ nd, :])),
                            "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP[ nd, :])),
                            "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION[ nd, :])),
                            "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN[ nd, :])),
                            "\t\t\tdice_similarity = {:.2f}".format( np.nanmean( DS[ nd, :]))]
            logger.info( ''.join( logging_str))
            f_csv.write( "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % ( data.img_list[ nd], pixel_error, np.nanmean( ACCURACY[ nd, :]), np.nanmean( TP[ nd, :]), np.nanmean( PRECISION[ nd, :]), np.nanmean( TN[ nd, :]), np.nanmean( DS[ nd, :])))

            if imgs is not None:
                img0 = imgs[ nd][ np.newaxis, ...]
                x0 = np.pad( img0, ( ( 0, 0), pad_shape0, pad_shape1, ( 0, 0)), mode = "constant")
            else:
                x0 = np.zeros( shape = ( ( y0.shape[ 0],) + ( self._input_HW) + ( 1,)), dtype = np.uint8)
            result_img_name, _ = os.path.splitext( data.img_list[ nd])
            data.save_prediction_img( output_img_path, result_img_name, x0, y0, pr, save_img_type = save_img_type, mask = None)
                
        logging_str = [ "total mean>>\n",
                        "\t\t\tpixel_error = {:.2f}%\n".format( total_pixel_error / data.num_examples),
                        "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY)),
                        "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP)),
                        "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION)),
                        "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN)),
                        "\t\t\tdice_similarity = {:.2f}".format( np.nanmean( DS))]
        logger.info( ''.join( logging_str))
        formatter = '[' + ( "{:6d}," * data.num_class)[ : -1] + ']'
        logging_str = [ "========================================================\n",
                        "\t\t\taccuracy = {0}\n".format( np.array2string( np.nanmean( ACCURACY, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                        "\t\t\trecall = {0}\n".format( np.array2string( np.nanmean( TP, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                        "\t\t\tprecision = {0}\n".format( np.array2string( np.nanmean( PRECISION, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                        "\t\t\ttrue_negatives = {0}\n".format( np.array2string( np.nanmean( TN, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                        "\t\t\tdice_similarity = {0}\n".format( np.array2string( np.nanmean( DS, axis = 0), max_line_width = 1000, precision = 4, separator = ',')),
                        "\t\t\tconfusion_matrix_nc>>\n",
                        *[ "\t\t\t\t%s\n" % ( formatter.format( *[ confusion_matrix_by_class[ nc1][ nc2] for nc2 in range( data.num_class)])) for nc1 in range( data.num_class)],
                        "\t\t\trecall_by_cm = {0}\n".format( np.array2string( np.array( [ confusion_matrix_by_class[ nc][ nc] / np.sum( confusion_matrix_by_class[ nc, :]) for nc in range( data.num_class)]), max_line_width = 1000, precision = 4, separator = ',')),
                        "\t\t\tprecision_by_cm = {0}\n".format( np.array2string( np.array( [ confusion_matrix_by_class[ nc][ nc] / np.sum( confusion_matrix_by_class[ :, nc]) for nc in range( data.num_class)]), max_line_width = 1000, precision = 4, separator = ','))]
        logger.info( ''.join( logging_str))

        f_csv.write( "\n\n")
        f_csv.write( "mean_nc_accuracy,%s\n" % ( np.array2string( np.nanmean( ACCURACY, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_nc_recall,%s\n" % ( np.array2string( np.nanmean( TP, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_nc_precision,%s\n" % ( np.array2string( np.nanmean( PRECISION, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_nc_dice_similarity,%s\n" % ( np.array2string( np.nanmean( DS, axis = 0), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "\n\n")
        f_csv.write( "mean_accuracy,%s\n" % ( np.array2string( np.nanmean( ACCURACY), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_recall,%s\n" % ( np.array2string( np.nanmean( TP), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_precision,%s\n" % ( np.array2string( np.nanmean( PRECISION), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_true_negatives,%s\n" % ( np.array2string( np.nanmean( TN), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "mean_dice_similarity,%s\n" % ( np.array2string( np.nanmean( DS), max_line_width = 1000, precision = 4, separator = ',')))
        f_csv.write( "\n\nconfusion_matrix_nc\n")
        f_csv.write( "%s\n" % ( np.array2string( confusion_matrix_by_class, max_line_width = 1000, separator = ',')))
        f_csv.close()


    def test_using_rank_one_vs_all( self, data, class_idx, min_rank, output_img_path, model_path, model_iter = -1, opt_kwargs = {}):

        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_test_using_rank_one_vs_all_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        use_average_mirror = opt_kwargs.pop( "use_average_mirror", False)
        min_threshold = opt_kwargs.pop( "min_threshold", 0)
        thresholds = opt_kwargs.pop( "thresholds", [])
        """
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_test_using_rank_one_vs_all_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        use_average_mirror = opt_kwargs.pop( "use_average_mirror", False)
        min_threshold = opt_kwargs.pop( "min_threshold", 0)
        thresholds = opt_kwargs.pop( "thresholds", [])

        if len( opt_kwargs):
            raise ValueError( "wrong opt_kwargs : %s" % ( str( opt_kwargs.keys())))

        if len( logger.handlers) == 2:
            logger.removeHandler( logger.handlers[ -1])

        if os.path.isdir( logging_folder) == False:
            os.makedirs( logging_folder)
        file_handler = logging.FileHandler( os.path.join( logging_folder, logging_name))
        file_handler.setFormatter( log_formatter)
        logger.addHandler( file_handler)

        logging_str = [ "test_parmas >>\n",
                        "\t\t\tlogging name : {0}\n".format( logging_name),
                        "\t\t\tdata_module : {0}\n".format( data.__module__),
                        "\t\t\tdata_path : {0}\n".format( data.dir),
                        "\t\t\tclass_idx : {0}\n".format( class_idx),
                        "\t\t\tmin_rank : {0}\n".format( min_rank),
                        "\t\t\toutput_img_path : {0}\n".format( output_img_path),
                        "\t\t\tmodel_path : {0}\n".format( model_path),
                        "\t\t\tuse_average_mirror : {0}\n".format( use_average_mirror),
                        "\t\t\tmin_threshold : {0}\n".format( min_threshold),
                        "\t\t\tthresholds : {0}\n".format( thresholds)]
        logger.info( ''.join( logging_str))
        
        if len( thresholds) == 0:
            TPs = None
            FPs = None
            FNs = None
            TNs = None
        else:
            TPs = np.zeros( shape = ( len( thresholds),), dtype = np.int32)
            FPs = np.zeros( shape = ( len( thresholds),), dtype = np.int32)
            FNs = np.zeros( shape = ( len( thresholds),), dtype = np.int32)
            TNs = np.zeros( shape = ( len( thresholds),), dtype = np.int32)

        shutil.rmtree( output_img_path, ignore_errors = True)
        time.sleep( 0.100)
        os.makedirs( output_img_path, exist_ok = True)

        with tf.Session() as sess:
                        
            sess.run( tf.global_variables_initializer())
            self.restore( sess, model_path, model_iter)

            pad_shape0 = ( ( self._input_HW[ 0] - self._output_HW[ 0]) // 2, ( self._input_HW[ 0] - self._output_HW[ 0]) // 2)
            pad_shape1 = ( ( self._input_HW[ 1] - self._output_HW[ 1]) // 2, ( self._input_HW[ 1] - self._output_HW[ 1]) // 2)
                
            data_num_class_wo_fake = data.num_class_wo_fake
            confusion_matrix_by_class = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
            for nd in range( data.num_examples):

                x0, y0, _, mask0 = data.get( pad_shape0, pad_shape1, nd)

                shape = y0.shape[ 1 : 3]

                x = np.ndarray( shape = ( 1,) + shape + ( data.num_channel,), dtype = np.float32)
                pr = np.zeros( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)
                acc_pr = np.zeros( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)
                gt = np.ndarray( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)

                step0 = np.arange( 0, shape[ 0], self._output_HW[ 0])
                ranges0 = []
                for ns in range( len( step0) - 1):
                    ranges0 = ranges0 + [ ( step0[ ns], step0[ ns + 1])]
                ranges0 = ranges0 + [ ( shape[ 0] - self._output_HW[ 0], shape[ 0])]

                step1 = np.arange( 0, shape[ 1], self._output_HW[ 1])
                ranges1 = []
                for ns in range( len( step1) - 1):
                    ranges1 = ranges1 + [ ( step1[ ns], step1[ ns + 1])]
                ranges1 = ranges1 + [ ( shape[ 1] - self._output_HW[ 1], shape[ 1])]

                window_rects = []
                for r0 in ranges0:
                    for r1 in ranges1:
                        window_rects += [ ( r0[ 0], r1[ 0], self._output_HW[ 0], self._output_HW[ 1])]


                if mask0 is not None:
                    new_window_rects = []
                    for nw, wr in enumerate( window_rects):
                        on_cnt = np.count_nonzero( mask0[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3]])
                        if on_cnt > np.prod( self._output_HW) * 0.1:
                            new_window_rects.append( wr)
                    window_rects = new_window_rects


                for nw, wr in enumerate( window_rects):

                    wx = x0[ :, wr[ 0] : wr[ 0] + wr[ 2] + np.sum( pad_shape0), wr[ 1] : wr[ 1] + wr[ 3] + np.sum( pad_shape1), :]
                    wy = y0[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :]
                    pr_ = sess.run( self._predictor, feed_dict = { self._x: wx, self._y: wy, self._keep_prob: 1.})
                    x[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] = wx[ :, pad_shape0[ 0] : -pad_shape0[ 1], pad_shape1[ 0] : -pad_shape1[ 1], :]
                    pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += pr_
                    acc_pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += 1
                    gt[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] = wy

                if use_average_mirror == True:

                    mirrored_x0s = [ np.flip( x0, axis = 1), np.flip( x0, axis = 2), np.flip( np.flip( x0, axis = 2), 1)]
                    mirrored_y0s = [ np.flip( y0, axis = 1), np.flip( y0, axis = 2), np.flip( np.flip( y0, axis = 2), 1)]
                    mirrored_prs = [ np.zeros_like( pr), np.zeros_like( pr), np.zeros_like( pr)]
                    for mirrored_x0, mirrored_y0, mirrored_pr in zip( mirrored_x0s, mirrored_y0s, mirrored_prs):
                        for nw, wr in enumerate( window_rects):
                            wx = mirrored_x0[ :, wr[ 0] : wr[ 0] + wr[ 2] + np.sum( pad_shape0), wr[ 1] : wr[ 1] + wr[ 3] + np.sum( pad_shape1), :]
                            wy = mirrored_y0[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :]
                            pr_ = sess.run( self._predictor, feed_dict = { self._x: wx, self._y: wy, self._keep_prob: 1.})
                            mirrored_pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += pr_
                    pr += np.flip( mirrored_prs[ 0], axis = 1)
                    pr += np.flip( mirrored_prs[ 1], axis = 2)
                    pr += np.flip( np.flip( mirrored_prs[ 2], axis = 1), axis = 2)
                    acc_pr = acc_pr + np.flip( acc_pr, axis = 1) + np.flip( acc_pr, axis = 2) + np.flip( np.flip( acc_pr, axis = 2), 1)
                pr = pr / acc_pr
                
                argmax_gt = np.argmax( gt, 3)
                argmax_gt_ncs = []
                for nc in range( data.num_class):
                    argmax_gt_ncs.append( argmax_gt == nc)
                argsort_pr = np.argsort( pr, axis = 3)[ :, :, :, :: -1]
                arg_pr_nc_where = np.where( argsort_pr == class_idx)
                nc_idx1 = arg_pr_nc_where[ 3] < min_rank
                
                for nt, threshold in enumerate( thresholds):
                    pr2 = np.copy( pr)
                    nc_idx2 = pr2[ arg_pr_nc_where[ 0], arg_pr_nc_where[ 1], arg_pr_nc_where[ 2], class_idx] > threshold
                    nc_idx = np.logical_and( nc_idx1, nc_idx2)
                    pr2[ arg_pr_nc_where[ 0][ nc_idx], arg_pr_nc_where[ 1][ nc_idx], arg_pr_nc_where[ 2][ nc_idx], class_idx] = 1.0
                    nc_idx = ~nc_idx
                    pr2[ arg_pr_nc_where[ 0][ nc_idx], arg_pr_nc_where[ 1][ nc_idx], arg_pr_nc_where[ 2][ nc_idx], class_idx] = 0.0

                    argmax_pr = np.argmax( pr2, 3)
                    argmax_pr_ncs = []
                    for nc in range( data.num_class):
                        argmax_pr_ncs.append( argmax_pr == nc)
                
                    confusion_matrix_by_class_ts = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
                    # confusion-matrix by class
                    for nc_gt in range( data.num_class):
                        for nc_pr in range( data.num_class):
                            cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                            confusion_matrix_by_class_ts[ nc_gt][ nc_pr] = cm_val
                    TP = confusion_matrix_by_class_ts[ class_idx, class_idx]
                    FP = np.sum( confusion_matrix_by_class_ts[ :, class_idx]) - TP
                    FN = np.sum( confusion_matrix_by_class_ts[ class_idx, :]) - TP
                    TN = np.sum( confusion_matrix_by_class_ts) - TP - FP - FN
                    TPs[ nt] += TP
                    FPs[ nt] += FP
                    FNs[ nt] += FN
                    TNs[ nt] += TN

                         
                pr2 = np.copy( pr)
                nc_idx2 = pr2[ arg_pr_nc_where[ 0], arg_pr_nc_where[ 1], arg_pr_nc_where[ 2], class_idx] > min_threshold
                nc_idx = np.logical_and( nc_idx1, nc_idx2)
                pr2[ arg_pr_nc_where[ 0][ nc_idx], arg_pr_nc_where[ 1][ nc_idx], arg_pr_nc_where[ 2][ nc_idx], class_idx] = 1.0
                nc_idx = ~nc_idx
                pr2[ arg_pr_nc_where[ 0][ nc_idx], arg_pr_nc_where[ 1][ nc_idx], arg_pr_nc_where[ 2][ nc_idx], class_idx] = 0.0

                argmax_pr = np.argmax( pr2, 3)
                argmax_pr_ncs = []
                for nc in range( data.num_class):
                    argmax_pr_ncs.append( argmax_pr == nc)
                
                confusion_matrix_by_class1 = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
                # confusion-matrix by class
                for nc_gt in range( data.num_class):
                    for nc_pr in range( data.num_class):
                        cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                        confusion_matrix_by_class1[ nc_gt][ nc_pr] = cm_val
                confusion_matrix_by_class = confusion_matrix_by_class1
                
                TP = confusion_matrix_by_class1[ class_idx, class_idx]
                FP = np.sum( confusion_matrix_by_class1[ :, class_idx]) - TP
                FN = np.sum( confusion_matrix_by_class1[ class_idx, :]) - TP
                TN = np.sum( confusion_matrix_by_class1) - TP - FP - FN
                logging_str = [ "image_name = {:}\n".format( data.img_list[ nd]),
                                "\t\t\tTP = {0}\n".format( TP),
                                "\t\t\tFP = {0}\n".format( FP),
                                "\t\t\tFN = {0}\n".format( FN),
                                "\t\t\tTN = {0}\n".format( TN),
                                "\t\t\trecall = {:.2f}\n".format( TP / ( TP + FN) if ( TP + FN) > 0 else 0),
                                "\t\t\tprecision = {:.2f}\n".format( TP / ( TP + FP) if ( TP + FP) > 0 else 0)]
                logger.info( ''.join( logging_str))

                x = np.pad( x, ( ( 0, 0), pad_shape0, pad_shape1, ( 0, 0)), mode = "constant")
                result_img_name, _ = os.path.splitext( data.img_list[ nd])
                data.save_prediction_img( output_img_path, result_img_name, x, gt, pr2, save_img_type = 1, mask = None)
                
            TP = confusion_matrix_by_class[ class_idx, class_idx]
            FP = np.sum( confusion_matrix_by_class[ :, class_idx]) - TP
            FN = np.sum( confusion_matrix_by_class[ class_idx, :]) - TP
            TN = np.sum( confusion_matrix_by_class) - TP - FP - FN
            logging_str = [ "total mean>>\n",
                            "\t\t\tTP = {0}\n".format( TP),
                            "\t\t\tFP = {0}\n".format( FP),
                            "\t\t\tFN = {0}\n".format( FN),
                            "\t\t\tTN = {0}\n".format( TN),
                            "\t\t\trecall = {:.2f}\n".format( TP / ( TP + FN) if ( TP + FN) > 0 else 0),
                            "\t\t\tprecision = {:.2f}\n".format( TP / ( TP + FP) if ( TP + FP) > 0 else 0)]
            logger.info( ''.join( logging_str))

        return TPs, FPs, FNs, TNs


    def get_response( self, img, tensor_name, model_path, iter = -1):
        
        with tf.Session() as sess:
            
            sess.run( tf.global_variables_initializer())
            self.restore( sess, model_path, iter)
            #output = tf.get_default_graph().get_tensor_by_name( tensor_name)
            response = sess.run( self._predictor, feed_dict = { self._x: img, self._keep_prob : np.float32( 1)})
            
            return response



    ################################################# util #################################################
    @property
    def num_layer( self):
        return self._num_layer


    @property
    def num_feature_root( self):
        return self._num_feature_root


    @property
    def num_channel( self):
        return self._num_channel


    @property
    def num_class( self):
        return self._num_class


    @property
    def input_HW( self):
        return self._input_HW


    def save( self, sess, model_path, error_rate, iter):
        temp = self._model_name + "_" + '%.2f' % error_rate
        save_path = self._saver.save( sess, os.path.join( model_path, temp), iter)
        return save_path
    

    def restore( self, sess, model_path, iter = -1):
        if type( iter) == int:
            if iter == -1: #last
                names = []
                iters = []
                [ ( names.append( name[ : -5]), iters.append( int( name[ name.rfind( '-') + 1 : -5]))) for name in os.listdir( model_path) if name.endswith( ".meta")]
                idx = np.argsort( iters)[ -1]
                riter = iters[ idx]
                restored_model_path = os.path.join( model_path, names[ idx])
                self._saver.restore( sess, restored_model_path)
            else:
                riter = iter
                names = [ name[ : -5] for name in os.listdir( model_path) if name.endswith( '-' + str( iter) + ".meta")]
                restored_model_path = os.path.join( model_path, names[ 0])
                self._saver.restore( sess, restored_model_path)
            logger.info("Model restored from file: %s" % restored_model_path)
        elif type( iter) == str:
            riter = int( iter.split( "-")[ -1])
            self._saver.restore( sess, iter)
            logger.info("Model restored from file: %s" % iter)
        else:
            raise ValueError( "iter must be type of str or int")
        return riter

        
    def output_verification_stats( self, sess, cost, use_weight_map, batch_x, batch_y, batch_weight):

        if use_weight_map == False:
            pr, loss = sess.run( [ self._predictor, cost], feed_dict = { self._x: batch_x, self._y: batch_y, self._keep_prob: 1.})
        else:
            pr, loss = sess.run( [ self._predictor, cost], feed_dict = { self._x: batch_x, self._y: batch_y, self._weight_map: batch_weight, self._keep_prob: 1.})
        
        error_rate = 100.0 - ( 100.0 * np.sum( np.argmax( pr, 3) == np.argmax( batch_y, 3)) / ( pr.shape[ 0] * pr.shape[ 1] * pr.shape[ 2]))
        logger.info( "Verification error= {:.2f}%, loss= {:.4f}".format( error_rate, loss))
        
        return pr, error_rate


    def output_minibatch_stats(self, sess, cost, use_weight_map, step, batch_x, batch_y, batch_weight):
        
        if use_weight_map == False:
            loss, pr = sess.run([ cost, self._predictor], feed_dict = { self._x: batch_x, self._y: batch_y, self._keep_prob: 1.})
        else:
            loss, pr = sess.run([ cost, self._predictor], feed_dict = { self._x: batch_x, self._y: batch_y, self._weight_map: batch_weight, self._keep_prob: 1.})

        error_rate = 100.0 - ( 100.0 * np.sum( np.argmax( pr, 3) == np.argmax( batch_y, 3)) / ( pr.shape[ 0] * pr.shape[ 1] * pr.shape[ 2]))
        logger.info( "Iter {:}, Minibatch Loss= {:.4f}, Minibatch error= {:.1f}%".format( step, loss, error_rate))

        return pr


    def output_test_stats( self, sess, cost, use_weight_map, data, use_average_mirror):

        pad_shape0 = ( 0, 0)
        pad_shape1 = ( 0, 0)
                
        total_pixel_error = 0.
        data_num_class_wo_fake = data.num_class_wo_fake
        ACCURACY = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        PRECISION = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        TP = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        TN = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        DS = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        confusion_matrix_by_class = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
        for nd in range( data.num_examples):

            x0, y0, weight0, mask0 = data.get( pad_shape0, pad_shape1, nd)

            shape = y0.shape[ 1 : 3]

            x = np.ndarray( shape = ( 1,) + shape + ( data.num_channel,), dtype = np.float32)
            pr = np.zeros( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)
            acc_pr = np.zeros( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)
            gt = np.ndarray( shape = ( 1,) + shape + ( data.num_class,), dtype = np.float32)
            total_loss = 0
            acc_loss_cnt = 0


            

            wx = x0
            wy = y0
            wweight = weight0
                
            if use_weight_map == False:
                pr_, loss = sess.run( [ self._predictor, cost], feed_dict = { self._x: wx, self._y: wy, self._keep_prob: 1.})
            else:
                pr_, loss = sess.run( [ self._predictor, cost], feed_dict = { self._x: wx, self._y: wy, self._weight_map: wweight, self._keep_prob: 1.})
            #x[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] = wx[ :, pad_shape0[ 0] : -pad_shape0[ 1], pad_shape1[ 0] : -pad_shape1[ 1], :]
            x = wx
            pr += pr_
            acc_pr += 1
            gt = wy
                
            total_loss += loss
            acc_loss_cnt += 1

            pr = pr / acc_pr
                
            argmax_pr = np.argmax( pr, 3)
            argmax_gt = np.argmax( gt, 3)
            argmax_pr_ncs = []
            argmax_gt_ncs = []
            for nc in range( data.num_class):
                argmax_pr_ncs.append( argmax_pr == nc)
                argmax_gt_ncs.append( argmax_gt == nc)
            for nc in range( data_num_class_wo_fake):
                argmax_pr_nc = argmax_pr_ncs[ nc]
                argmax_gt_nc = argmax_gt_ncs[ nc]
                tp = np.count_nonzero( np.logical_and( argmax_pr_nc, argmax_gt_nc))
                tn = np.count_nonzero( np.logical_and( ( ~argmax_pr_nc), ( ~argmax_gt_nc)))
                union = np.count_nonzero( np.logical_or( argmax_pr_nc, argmax_gt_nc))
                tp_fp = np.count_nonzero( argmax_pr_nc)
                tp_fn = np.count_nonzero( argmax_gt_nc)
                not_tp_fn = np.count_nonzero( ~argmax_gt_nc)

                ACCURACY[ nd, nc] = ( tp / tp_fp) if tp_fp > 0 else np.nan
                PRECISION[ nd, nc] = ( tp / union) if union > 0 else np.nan
                TP[ nd, nc] = ( tp / tp_fn) if tp_fn > 0 else np.nan
                TN[ nd, nc] = ( tn / not_tp_fn) if not_tp_fn > 0 else np.nan
                DS[ nd, nc] = ( 2 * tp / ( tp_fp + tp_fn)) if tp_fp + tp_fn > 0 else np.nan
            # confusion-matrix by class
            for nc_gt in range( data.num_class):
                for nc_pr in range( data.num_class):
                    cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                    confusion_matrix_by_class[ nc_gt][ nc_pr] += cm_val

            pixel_error = 100.0 * np.count_nonzero( argmax_pr != argmax_gt) / ( 1 * pr.shape[ 1] * pr.shape[ 2])
            total_pixel_error += pixel_error
           
        formatter = '[' + ( "{:6d}," * data.num_class)[ : -1] + ']'
        logging_str = [ "Test error>>\n",
                        "\t\t\tloss = {:.2f}\n".format( total_loss / acc_loss_cnt),
                        "\t\t\tpixel_error = {:.2f}%\n".format( total_pixel_error / data.num_examples),
                        "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY)),
                        "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP)),
                        "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION)),
                        "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN)),
                        "\t\t\tdice_similarity = {:.2f}\n".format( np.nanmean( DS)),
                        "\t\t\tconfusion_matrix_nc>>\n",
                        *[ "\t\t\t\t%s\n" % ( formatter.format( *[ confusion_matrix_by_class[ nc1][ nc2] for nc2 in range( data.num_class)])) for nc1 in range( data.num_class)]]
                        #*[ "\t\t\t\t%s\n" % ( np.array2string( confusion_matrix_by_class[ nc][ :], max_line_width = 1000, separator = ',')) for nc in range( data.num_class)]]
                        
        logger.info( ''.join( logging_str))
        return total_pixel_error / data.num_examples, confusion_matrix_by_class