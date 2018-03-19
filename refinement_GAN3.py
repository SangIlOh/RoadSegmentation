## INPUT of generator: prediction result of semantic segmentation from Deeplab
## OUTPUT of generator: refined semantic segmentation of INPUT

## INPUT of discriminator: OUTPUT of generator & training dataset
## OUTPUT of discriminator: real or fake

## training procedure:
    ## 1. train DeepLab
    ## 2. train GAN with 1's DeepLab
    ## 3. train 1 & 2 -> to fine-tune DeepLab
import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time
import cv2

log_formatter = logging.Formatter( "[%(asctime)s] %(message)s", datefmt = "%m-%d %H:%M:%S")
logger = logging.getLogger( "model_refiner.py")
logger.setLevel( logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter( log_formatter)
logger.addHandler( console_handler)

def find_nth( haystack, needle, n):
        start = haystack.find( needle)
        while start >= 0 and n > 1:
            start = haystack.find( needle, start + len( needle))
            n -= 1
        return start

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def conv2d( x, W, stride = 1, name = None):

    return tf.nn.conv2d( x, W, strides = [ 1, stride, stride, 1], padding = "VALID", name = name)

def seperable_conv2d(x, depth_filter, point_filter, name = None):

    return tf.nn.separable_conv2d(x, depthwise_filter = depth_filter, pointwise_filter = point_filter, strides = [ 1, 2, 2, 1], padding = "SAME", name = name)

def max_pool( x, n):
    return tf.nn.max_pool( x, ksize = [ 1, n, n, 1], strides = [ 1, n, n, 1], padding = "VALID")

def crop_and_concat( x1, x2, shape1, shape2):
    # offsets for the top left corner of the crop
    offsets = [0, ( shape1[ 0] - shape2[ 0]) // 2, ( shape1[ 1] - shape2[ 1]) // 2, 0]
    size = [-1, shape2[ 0], shape2[ 1], -1]
    x1_crop = tf.slice( x1, offsets, size)
    return tf.concat( [ x1_crop, x2], 3)   

def deconv2d( x, W, stride):
    x_shape = tf.shape( x)
    output_shape = tf.stack( [ x_shape[ 0], x_shape[ 1] * 2, x_shape[ 2] * 2, x_shape[ 3] // 2])
    return tf.nn.conv2d_transpose( x, W, output_shape, strides = [ 1, stride, stride, 1], padding = "SAME")

def seperable_deconv2d(x, depth_filter, point_filter, name = None):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    rx = tf.image.resize_images(x, [h*2, w*2], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.nn.separable_conv2d(rx, depthwise_filter = depth_filter, pointwise_filter = point_filter, strides = [ 1, 1, 1, 1], padding = "SAME", name = name)

def conv2d_with_dropout( x, W, keep_prob, name = None):
    conv_2d = tf.nn.conv2d( x, W, strides = [ 1, 1, 1, 1], padding = "VALID")
    return tf.nn.dropout( conv_2d, keep_prob, name = name)

def weight_variable( name, shape, mean = 0.0, stddev = 1.0):
    initial = tf.truncated_normal( shape = shape, dtype = tf.float32, mean = mean, stddev = stddev)
    return tf.Variable( initial, name = name)

def weight_variable_deconv( name, shape):
    width = shape[ 0]
    height = shape[ 1]
    f = np.ceil( width / 2.0)
    c = ( 2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros( [ shape[ 0], shape[ 1]])
    for x in range( width):
        for y in range( height):
            value = ( 1 - abs( x / f - c)) * ( 1 - abs( y / f - c))
            bilinear[ x, y] = value
    bilinear /= shape[ 3]
    initial = np.transpose( np.tile( bilinear, [ shape[ 2], shape[ 3], 1, 1]), [ 2, 3, 0, 1])
    noise = np.random.uniform( -( np.min( bilinear) / 10), np.min( bilinear / 10), size = initial.shape)
    return tf.Variable( initial + noise, name = name, dtype = np.float32)

def bias_variable( name, shape, value = 0.1):
    initial = tf.constant( value, shape = shape, dtype = tf.float32)
    return tf.Variable( initial, name = name)

def build_generator(inputs, is_training, weights, biases, keep_prob, num_channel, output_HW):
    ## unet params
    num_layer = 7
    num_feature_root = 64
    num_class = num_channel
    input_HW = output_HW

    filter_size = 4
    depth_multiplier = 1
    layers = []

    ###### ENCODER #####
    with tf.variable_scope("0_down"):
        feat_n = num_feature_root
        depth_filter = weight_variable( "Wd_conv1", shape = [ filter_size, filter_size, num_channel, depth_multiplier], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * num_channel)))
        point_filter = weight_variable( "Wp_conv1", shape = [ 1, 1, depth_multiplier*num_channel, feat_n], stddev = np.math.sqrt( 2.0 / ( depth_multiplier*num_channel)))
        conv1 = seperable_conv2d(inputs, depth_filter, point_filter)

        layers.append(conv1)
        weights.append(depth_filter)
        weights.append(point_filter)

    efn_list = [num_feature_root*2, num_feature_root*4, num_feature_root*8]# feature number list of encoder
    for nl in range(num_layer):
        with tf.variable_scope( "%d_down" % (nl + 1)):
            if nl < 3:
                depth_filter = weight_variable( "Wd_conv1", shape = [ filter_size, filter_size, feat_n, depth_multiplier], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * feat_n)))
                point_filter = weight_variable( "Wp_conv1", shape = [ 1, 1, depth_multiplier*feat_n, efn_list[nl]], stddev = np.math.sqrt( 2.0 / ( depth_multiplier*feat_n)))
                feat_n = efn_list[nl]
            else:
                depth_filter = weight_variable( "Wd_conv1", shape = [ filter_size, filter_size, feat_n, depth_multiplier], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * feat_n)))
                point_filter = weight_variable( "Wp_conv1", shape = [ 1, 1, depth_multiplier*feat_n, efn_list[-1]], stddev = np.math.sqrt( 2.0 / ( depth_multiplier*feat_n)))
                feat_n = efn_list[-1]
        
            h_conv1 = lrelu(layers[-1], 0.2)
            conv1 = seperable_conv2d(h_conv1, depth_filter, point_filter, name = "conv1")
            bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training, name = "bn_conv1")

            layers.append(bn_conv1)
            weights.append(depth_filter)
            weights.append(point_filter)

    ###### DECODER #####
    dfn_list = [num_feature_root*8*2, num_feature_root*4*2, num_feature_root*2*2, num_feature_root*2]# feature number list of decoder
    num_encoder = len(layers)
    for nl in range(num_layer):
        skip_idx = num_encoder - nl - 1
        with tf.variable_scope( "%d_up" % (nl + 1)):
            if nl == 0:
                decoder_input = tf.nn.relu(layers[-1])
            else:
                htmp = layers[-1].get_shape()[1].value
                wtmp = layers[-1].get_shape()[2].value
                skip_out = tf.image.resize_images(layers[skip_idx], [htmp, wtmp], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                decoder_input = tf.nn.relu(tf.concat([layers[-1], skip_out], axis = 3))

            input_channel = decoder_input.get_shape()[3].value
            if nl < 4:
                depth_filter = weight_variable( "Wd_deconv1", shape = [ filter_size, filter_size, input_channel, depth_multiplier], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * input_channel)))
                point_filter = weight_variable( "Wp_deconv1", shape = [ 1, 1, depth_multiplier*input_channel, dfn_list[nl]], stddev = np.math.sqrt( 2.0 / ( depth_multiplier*input_channel)))
            else:
                depth_filter = weight_variable( "Wd_deconv1", shape = [ filter_size, filter_size, input_channel, depth_multiplier], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * input_channel)))
                point_filter = weight_variable( "Wp_deconv1", shape = [ 1, 1, depth_multiplier*input_channel, dfn_list[-1]], stddev = np.math.sqrt( 2.0 / ( depth_multiplier*input_channel)))

            deconv1 = seperable_deconv2d(decoder_input, depth_filter, point_filter, name = "deconv1")
            bn_deconv1 = tf.layers.batch_normalization(deconv1, training=is_training, name = "bn_deconv1")
            if nl < 3:
                bn_deconv1 = tf.nn.dropout(bn_deconv1, keep_prob = 0.5)

            layers.append(bn_deconv1)
            weights.append(depth_filter)
            weights.append(point_filter)
    
    with tf.variable_scope("last_up"):
        htmp = layers[-1].get_shape()[1].value
        wtmp = layers[-1].get_shape()[2].value
        skip_out = tf.image.resize_images(layers[0], [htmp, wtmp], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        decoder_input = tf.nn.relu(tf.concat([layers[-1], skip_out], axis = 3))

        input_channel = decoder_input.get_shape()[3].value
        depth_filter = weight_variable( "Wd_deconv1", shape = [ filter_size, filter_size, input_channel, depth_multiplier], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * input_channel)))
        point_filter = weight_variable( "Wp_deconv1", shape = [ 1, 1, depth_multiplier*input_channel, num_class], stddev = np.math.sqrt( 2.0 / ( depth_multiplier*input_channel)))
        output_map = seperable_deconv2d(decoder_input, depth_filter, point_filter, name = "deconv1")
        output_map = tf.image.resize_bilinear(tf.tanh(output_map), [output_HW[0], output_HW[1]], name = "output_map")

        layers.append(output_map)
        weights.append(depth_filter)
        weights.append(point_filter)

    return output_map, weights, biases


def build_discriminator(inputs, targets, weights, is_training):
    num_layer = 4
    num_feature_root = 64

    input = tf.concat([inputs, targets], axis = 3)

    convs = []
    
    with tf.variable_scope( "layer_disc_1"):
        w1 = weight_variable( "W_conv1", shape = [ 4, 4, input.get_shape()[3].value, num_feature_root], stddev = np.math.sqrt( 2.0 / ( ( 4 ** 2) * ( num_feature_root))))
        conv1 = conv2d( tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), w1, stride = 2, name = "conv1")
        h_conv1 = lrelu(conv1, 0.2)
        weights.append(w1)
        convs.append(h_conv1)

    feature_num = num_feature_root
    for layer_scope_id in range( 2, num_layer+1, 1):
        with tf.variable_scope( "layer_disc_%d" % layer_scope_id):
            feature_num = feature_num * 2
            stride = 1 if layer_scope_id == num_layer else 2

            w1 = weight_variable( "W_conv1", shape = [ 4, 4, np.uint16(feature_num/2), feature_num], stddev = np.math.sqrt( 2.0 / ( ( 4 ** 2) * ( np.uint16(feature_num/2)))))
            conv1 = conv2d(tf.pad(convs[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), w1, stride = stride, name = "conv1")
            bn_conv1 = tf.layers.batch_normalization(conv1, axis=3, epsilon=1e-5, momentum=0.1, training=is_training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
            h_conv1 = lrelu(bn_conv1, 0.2)
            weights.append(w1)
            convs.append(h_conv1)

    with tf.variable_scope("layer_disc_5"):
        input_conv = tf.pad(convs[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        w1 = weight_variable( "W_conv1", shape = [ 4, 4, input_conv.get_shape()[-1].value, 1], stddev = np.math.sqrt( 2.0 / ( ( 4 ** 2) * ( input_conv.get_shape()[-1].value))))
        conv1 = conv2d(input_conv, w1, stride = 1, name = "conv1")
        output = tf.sigmoid(conv1)
        weights.append(w1)
        convs.append(output)
        

    return convs[-1], conv1, weights

class build_refine_model(object):
    _model_name = "refine_GAN"
    def __init__(self, num_channel, num_class, output_HW): # inputs: segmentation result from the first network, targets: final segmentation result

        self._output_HW = output_HW
        self._input_HW = output_HW
        self._num_channel = num_channel
        self._num_class = num_class
        self._weights_gan = []
        self._biases_gan = []
        # graph for DeepLab
        with tf.name_scope( "deeplab_v2"):
            # input for DeepLab
            with tf.name_scope( "input"):
                self._x = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_HW[0], self._input_HW[ 1], self._num_channel], name = "x")
                self._y = tf.placeholder( dtype = tf.float32, shape = [ None, output_HW[ 0], output_HW[ 1], self._num_class], name = "y")
                self._is_trains = tf.placeholder( dtype = tf.bool, shape = (), name = "is_trains")

            self._weights = []
            self._biases = []
            with tf.name_scope( "graph"):
                                
                with tf.name_scope( "layer0"):
                    scope_name = tf.contrib.framework.get_name_scope()
                    w_conv1_0 = weight_variable( "W_conv1_0", shape = [ 7, 7, self._num_channel, 64], stddev = np.math.sqrt( 2.0 / ( 7 * 7 * self._num_channel)))
                    conv1 = tf.nn.conv2d( self._x, w_conv1_0, strides = [ 1, 2, 2, 1], padding = "VALID", name = "conv1")
                    bn_conv1 = tf.layers.batch_normalization(conv1, training=self._is_trains, name = scope_name + "bn_conv1" )
                    bn_conv1 = tf.nn.relu( bn_conv1, name = "bn_conv1")
                    pool1 = tf.nn.max_pool( bn_conv1, ksize = [ 1, 3, 3, 1], strides = [ 1, 2, 2, 1], padding = "VALID") # yn

                    w_conv2_0 = weight_variable( "W_conv2_0", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2a_branch1 = tf.nn.conv2d( pool1, w_conv2_0, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch1")
                    bn2a_branch1 = tf.layers.batch_normalization(res2a_branch1, training=self._is_trains, name = scope_name + "bn2a_branch1")

                    self._weights.append( w_conv1_0)
                    self._weights.append( w_conv2_0)
                    
                with tf.name_scope( "layer1"):
                    scope_name = tf.contrib.framework.get_name_scope()
                    w_conv1_1 = weight_variable( "W_conv1_1", shape = [ 1, 1, 64, 64], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2a_branch2a = tf.nn.conv2d( pool1, w_conv1_1, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch2a")
                    bn2a_branch2a = tf.layers.batch_normalization(res2a_branch2a, training=self._is_trains, name = scope_name + "bn2a_branch2a")
                    bn2a_branch2a = tf.nn.relu( bn2a_branch2a, name = "bn2a_branch2a")

                    w_conv2_1 = weight_variable( "W_conv2_1", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)))
                    res2a_branch2b = tf.nn.conv2d( bn2a_branch2a, w_conv2_1, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch2b")
                    bn2a_branch2b = tf.layers.batch_normalization(res2a_branch2b, training=self._is_trains, name = scope_name + "bn2a_branch2b")
                    bn2a_branch2b = tf.nn.relu( bn2a_branch2b, name = "bn2a_branch2b")

                    w_conv3_1 = weight_variable( "W_conv3_1", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2a_branch2c = tf.nn.conv2d( bn2a_branch2b, w_conv3_1, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2a_branch2c")
                    bn2a_branch2c = tf.layers.batch_normalization(res2a_branch2c, training=self._is_trains, name = scope_name + "bn2a_branch2c")

                    self._weights.append( w_conv1_1)
                    self._weights.append( w_conv2_1)
                    self._weights.append( w_conv3_1)

                with tf.name_scope( "layer_2" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res2a = tf.add(bn2a_branch1, tf.image.resize_bilinear(bn2a_branch2c, [bn2a_branch1.get_shape()[1].value, bn2a_branch1.get_shape()[2].value,]), name = "res2a")
                    res2a_relu = tf.nn.relu( res2a, name = "res2a_relu")

                    w_conv1_2 = weight_variable( "W_conv1_2", shape = [ 1, 1, 256, 64], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res2b_branch2a = tf.nn.conv2d( res2a_relu, w_conv1_2, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2b_branch2a")
                    bn2b_branch2a = tf.layers.batch_normalization(res2b_branch2a, training=self._is_trains, name = scope_name + "bn2b_branch2a")
                    bn2b_branch2a = tf.nn.relu( bn2b_branch2a, name = "bn2b_branch2a")

                    w_conv2_2 = weight_variable( "W_conv2_2", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)))
                    res2b_branch2b = tf.nn.conv2d( bn2b_branch2a, w_conv2_2, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2b_branch2b")
                    bn2b_branch2b = tf.layers.batch_normalization(res2b_branch2b, training=self._is_trains, name = scope_name + "bn2b_branch2b")
                    bn2b_branch2b = tf.nn.relu( bn2b_branch2b, name = "bn2b_branch2b")

                    w_conv3_2 = weight_variable( "W_conv3_2", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2b_branch2c = tf.nn.conv2d( bn2b_branch2b, w_conv3_2, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2b_branch2c")
                    bn2b_branch2c = tf.layers.batch_normalization(res2b_branch2c, training=self._is_trains, name = scope_name + "bn2b_branch2c")

                    self._weights.append( w_conv1_2)
                    self._weights.append( w_conv2_2)
                    self._weights.append( w_conv3_2)

                with tf.name_scope( "layer_3" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res2b = tf.add(res2a_relu, tf.image.resize_bilinear(bn2b_branch2c, [res2a_relu.get_shape()[1].value, res2a_relu.get_shape()[2].value,]), name = "res2b")
                    res2b_relu = tf.nn.relu( res2b, name = "res2b_relu")

                    w_conv1_3 = weight_variable( "W_conv1_3", shape = [ 1, 1, 256, 64], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res2c_branch2a = tf.nn.conv2d( res2b_relu, w_conv1_3, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2c_branch2a")
                    bn2c_branch2a = tf.layers.batch_normalization(res2c_branch2a, training=self._is_trains, name = scope_name + "bn2c_branch2a")
                    bn2c_branch2a = tf.nn.relu( bn2c_branch2a, name = "bn2c_branch2a")

                    w_conv2_3 = weight_variable( "W_conv2_3", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)))
                    res2c_branch2b = tf.nn.conv2d( bn2c_branch2a, w_conv2_3, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2c_branch2b")
                    bn2c_branch2b = tf.layers.batch_normalization(res2c_branch2b, training=self._is_trains, name = scope_name + "bn2c_branch2b")
                    bn2c_branch2b = tf.nn.relu( bn2c_branch2b, name = "bn2c_branch2b")

                    w_conv3_3 = weight_variable( "W_conv3_3", shape = [ 1, 1, 64, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 64)))
                    res2c_branch2c = tf.nn.conv2d( bn2c_branch2b, w_conv3_3, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res2c_branch2c")
                    bn2c_branch2c = tf.layers.batch_normalization(res2c_branch2c, training=self._is_trains, name = scope_name + "bn2c_branch2c")

                    self._weights.append( w_conv1_3)
                    self._weights.append( w_conv2_3)
                    self._weights.append( w_conv3_3)

                with tf.name_scope( "layer_4" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res2c = tf.add(res2b_relu, tf.image.resize_bilinear(bn2c_branch2c, [res2b_relu.get_shape()[1].value, res2b_relu.get_shape()[2].value,]), name = "res2c")
                    res2c_relu = tf.nn.relu( res2c, name = "res2c_relu")

                    w_conv1_4 = weight_variable( "W_conv1_4", shape = [ 1, 1, 256, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res3a_branch1 = tf.nn.conv2d( res2c_relu, w_conv1_4, strides = [ 1, 2, 2, 1], padding = "VALID", name = "res3a_branch1")
                    bn3a_branch1 = tf.layers.batch_normalization(res3a_branch1, training=self._is_trains, name = scope_name + "")

                    self._weights.append( w_conv1_4)

                with tf.name_scope( "layer_5" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    w_conv1_5 = weight_variable( "W_conv1_5", shape = [ 1, 1, 256, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3a_branch2a = tf.nn.conv2d( res2c_relu, w_conv1_5, strides = [ 1, 2, 2, 1], padding = "VALID", name = "res3a_branch2a")
                    bn3a_branch2a = tf.layers.batch_normalization(res3a_branch2a, training=self._is_trains, name = scope_name + "bn3a_branch2a")
                    bn3a_branch2a = tf.nn.relu( bn3a_branch2a, name = "bn3a_branch2a")

                    w_conv2_5 = weight_variable( "W_conv2_5", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3a_branch2b = tf.nn.conv2d( bn3a_branch2a, w_conv2_5, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3a_branch2b")
                    bn3a_branch2b = tf.layers.batch_normalization(res3a_branch2b, training=self._is_trains, name = scope_name + "bn3a_branch2b")
                    bn3a_branch2b = tf.nn.relu( bn3a_branch2b, name = "bn3a_branch2b")

                    w_conv3_5 = weight_variable( "W_conv3_5", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3a_branch2c = tf.nn.conv2d( bn3a_branch2b, w_conv3_5, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3a_branch2c")
                    bn3a_branch2c = tf.layers.batch_normalization(res3a_branch2c, training=self._is_trains, name = scope_name + "bn3a_branch2c")

                    self._weights.append( w_conv1_5)
                    self._weights.append( w_conv2_5)
                    self._weights.append( w_conv3_5)

                with tf.name_scope( "layer_6" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res3a = tf.add(bn3a_branch1, tf.image.resize_bilinear(bn3a_branch2c, [bn3a_branch1.get_shape()[1].value, bn3a_branch1.get_shape()[2].value,]), name = "res3a")
                    res3a_relu = tf.nn.relu( res3a, name = "res3a_relu")

                    w_conv1_6 = weight_variable( "W_conv1_6", shape = [ 1, 1, 512, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3b1_branch2a = tf.nn.conv2d( res3a_relu, w_conv1_6, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b1_branch2a")
                    bn3b1_branch2a = tf.layers.batch_normalization(res3b1_branch2a, training=self._is_trains, name = scope_name + "bn3b1_branch2a")
                    bn3b1_branch2a = tf.nn.relu( bn3b1_branch2a, name = "bn3b1_branch2a")

                    w_conv2_6 = weight_variable( "W_conv2_6", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3b1_branch2b = tf.nn.conv2d( bn3b1_branch2a, w_conv2_6, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b1_branch2b")
                    bn3b1_branch2b = tf.layers.batch_normalization(res3b1_branch2b, training=self._is_trains, name = scope_name + "bn3b1_branch2b")
                    bn3b1_branch2b = tf.nn.relu( bn3b1_branch2b, name = "bn3b1_branch2b")

                    w_conv3_6 = weight_variable( "W_conv3_6", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3b1_branch2c = tf.nn.conv2d( bn3b1_branch2b, w_conv3_6, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b1_branch2c")
                    bn3b1_branch2c = tf.layers.batch_normalization(res3b1_branch2c, training=self._is_trains, name = scope_name + "bn3b1_branch2c")

                    self._weights.append( w_conv1_6)
                    self._weights.append( w_conv2_6)
                    self._weights.append( w_conv3_6)

                with tf.name_scope( "layer_7" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res3b1 = tf.add(res3a_relu, tf.image.resize_bilinear(bn3b1_branch2c, [res3a_relu.get_shape()[1].value, res3a_relu.get_shape()[2].value,]), name = "res3b1")
                    res3b1_relu = tf.nn.relu( res3b1, name = "res3b1_relu")

                    w_conv1_7 = weight_variable( "W_conv1_7", shape = [ 1, 1, 512, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3b2_branch2a = tf.nn.conv2d( res3b1_relu, w_conv1_7, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b2_branch2a")
                    bn3b2_branch2a = tf.layers.batch_normalization(res3b2_branch2a, training=self._is_trains, name = scope_name + "bn3b2_branch2a")
                    bn3b2_branch2a = tf.nn.relu( bn3b2_branch2a, name = "bn3b2_branch2a")

                    w_conv2_7 = weight_variable( "W_conv2_7", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3b2_branch2b = tf.nn.conv2d( bn3b2_branch2a, w_conv2_7, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b2_branch2b")
                    bn3b2_branch2b = tf.layers.batch_normalization(res3b2_branch2b, training=self._is_trains, name = scope_name + "bn3b2_branch2b")
                    bn3b2_branch2b = tf.nn.relu( bn3b1_branch2b, name = "bn3b2_branch2b")

                    w_conv3_7 = weight_variable( "W_conv3_7", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3b2_branch2c = tf.nn.conv2d( bn3b2_branch2b, w_conv3_7, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b2_branch2c")
                    bn3b2_branch2c = tf.layers.batch_normalization(res3b2_branch2c, training=self._is_trains, name = scope_name + "bn3b2_branch2c")

                    self._weights.append( w_conv1_7)
                    self._weights.append( w_conv2_7)
                    self._weights.append( w_conv3_7)

                with tf.name_scope( "layer_8" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res3b2 = tf.add(res3b1_relu, tf.image.resize_bilinear(bn3b2_branch2c, [res3b1_relu.get_shape()[1].value, res3b1_relu.get_shape()[2].value,]), name = "res3b2")
                    res3b2_relu = tf.nn.relu( res3b2, name = "res3b2_relu")

                    w_conv1_8 = weight_variable( "W_conv1_8", shape = [ 1, 1, 512, 128], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res3b3_branch2a = tf.nn.conv2d( res3b2_relu, w_conv1_8, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b3_branch2a")
                    bn3b3_branch2a = tf.layers.batch_normalization(res3b3_branch2a, training=self._is_trains, name = scope_name + "bn3b3_branch2a")
                    bn3b3_branch2a = tf.nn.relu( bn3b2_branch2a, name = "bn3b3_branch2a")

                    w_conv2_8 = weight_variable( "W_conv2_8", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)))
                    res3b3_branch2b = tf.nn.conv2d( bn3b3_branch2a, w_conv2_8, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b3_branch2b")
                    bn3b3_branch2b = tf.layers.batch_normalization(res3b3_branch2b, training=self._is_trains, name = scope_name + "bn3b3_branch2b")
                    bn3b3_branch2b = tf.nn.relu( bn3b3_branch2b, name = "bn3b3_branch2b")

                    w_conv3_8 = weight_variable( "W_conv3_8", shape = [ 1, 1, 128, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 128)))
                    res3b3_branch2c = tf.nn.conv2d( bn3b3_branch2b, w_conv3_8, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res3b3_branch2c")
                    bn3b3_branch2c = tf.layers.batch_normalization(res3b3_branch2c, training=self._is_trains, name = scope_name + "bn3b3_branch2c")

                    self._weights.append( w_conv1_8)
                    self._weights.append( w_conv2_8)
                    self._weights.append( w_conv3_8)

                with tf.name_scope( "layer_9" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res3b3 = tf.add(res3b2_relu, tf.image.resize_bilinear(bn3b3_branch2c, [res3b2_relu.get_shape()[1].value, res3b2_relu.get_shape()[2].value,]), name = "res3b3")
                    res3b3_relu = tf.nn.relu( res3b3, name = "res3b3_relu")

                    w_conv1_9 = weight_variable( "W_conv1_9", shape = [ 1, 1, 512, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res4a_branch1 = tf.nn.conv2d( res3b3_relu, w_conv1_9, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4a_branch1")
                    bn4a_branch1 = tf.layers.batch_normalization(res4a_branch1, training=self._is_trains, name = scope_name + "bn4a_branch1")

                    self._weights.append( w_conv1_9)

                with tf.name_scope( "layer_10" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    w_conv1_10 = weight_variable( "W_conv1_10", shape = [ 1, 1, 512, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res4a_branch2a = tf.nn.conv2d( res3b3_relu, w_conv1_10, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4a_branch2a")
                    bn4a_branch2a = tf.layers.batch_normalization(res4a_branch2a, training=self._is_trains, name = scope_name + "bn4a_branch2a")
                    bn4a_branch2a = tf.nn.relu( bn4a_branch2a, name = "bn4a_branch2a")

                    w_conv2_10 = weight_variable( "W_conv2_10", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4a_branch2b = tf.nn.atrous_conv2d( bn4a_branch2a, w_conv2_10, rate = 2, padding = "SAME", name = "res4a_branch2b")
                    bn4a_branch2b = tf.layers.batch_normalization(res4a_branch2b, training=self._is_trains, name = scope_name + "bn4a_branch2b")
                    bn4a_branch2b = tf.nn.relu( bn4a_branch2b, name = "bn4a_branch2b")

                    w_conv3_10 = weight_variable( "W_conv3_10", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4a_branch2c = tf.nn.conv2d( bn4a_branch2b, w_conv3_10, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4a_branch2c")
                    bn4a_branch2c = tf.layers.batch_normalization(res4a_branch2c, training=self._is_trains, name = scope_name + "bn4a_branch2c")

                    self._weights.append( w_conv1_10)
                    self._weights.append( w_conv2_10)
                    self._weights.append( w_conv3_10)

                with tf.name_scope( "layer_11" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4a = tf.add(bn4a_branch1, tf.image.resize_bilinear(bn4a_branch2c, [bn4a_branch1.get_shape()[1].value, bn4a_branch1.get_shape()[2].value,]), name = "res4a")
                    res4a_relu = tf.nn.relu( res4a, name = "res4a_relu")

                    w_conv1_11 = weight_variable( "W_conv1_11", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b1_branch2a = tf.nn.conv2d( res4a_relu, w_conv1_11, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b1_branch2a")
                    bn4b1_branch2a = tf.layers.batch_normalization(res4b1_branch2a, training=self._is_trains, name = scope_name + "bn4b1_branch2a")
                    bn4b1_branch2a = tf.nn.relu( bn4b1_branch2a, name = "bn4b1_branch2a")

                    w_conv2_11 = weight_variable( "W_conv2_11", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b1_branch2b = tf.nn.atrous_conv2d( bn4b1_branch2a, w_conv2_11, rate = 2, padding = "SAME", name = "res4b1_branch2b")
                    bn4b1_branch2b = tf.layers.batch_normalization(res4b1_branch2b, training=self._is_trains, name = scope_name + "bn4b1_branch2b")
                    bn4b1_branch2b = tf.nn.relu( bn4b1_branch2b, name = "bn4b1_branch2b")

                    w_conv3_11 = weight_variable( "W_conv3_11", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b1_branch2c = tf.nn.conv2d( bn4b1_branch2b, w_conv3_11, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b1_branch2c")
                    bn4b1_branch2c = tf.layers.batch_normalization(res4b1_branch2c, training=self._is_trains, name = scope_name + "bn4b1_branch2c")

                    self._weights.append( w_conv1_11)
                    self._weights.append( w_conv2_11)
                    self._weights.append( w_conv3_11)

                with tf.name_scope( "layer_12" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b1 = tf.add(res4a_relu, tf.image.resize_bilinear(bn4b1_branch2c, [res4a_relu.get_shape()[1].value, res4a_relu.get_shape()[2].value,]), name = "res4b1")
                    res4b1_relu = tf.nn.relu( res4b1, name = "res4b1_relu")

                    w_conv1_12 = weight_variable( "W_conv1_12", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b2_branch2a = tf.nn.conv2d( res4b1_relu, w_conv1_12, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b2_branch2a")
                    bn4b2_branch2a = tf.layers.batch_normalization(res4b2_branch2a, training=self._is_trains, name = scope_name + "bn4b2_branch2a")
                    bn4b2_branch2a = tf.nn.relu( bn4b1_branch2a, name = "bn4b2_branch2a")

                    w_conv2_12 = weight_variable( "W_conv2_12", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b2_branch2b = tf.nn.atrous_conv2d( bn4b2_branch2a, w_conv2_12, rate = 2, padding = "SAME", name = "res4b2_branch2b")
                    bn4b2_branch2b = tf.layers.batch_normalization(res4b2_branch2b, training=self._is_trains, name = scope_name + "bn4b2_branch2b")
                    bn4b2_branch2b = tf.nn.relu( bn4b2_branch2b, name = "bn4b2_branch2b")

                    w_conv3_12 = weight_variable( "W_conv3_12", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b2_branch2c = tf.nn.conv2d( bn4b2_branch2b, w_conv3_12, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b2_branch2c")
                    bn4b2_branch2c = tf.layers.batch_normalization(res4b2_branch2c, training=self._is_trains, name = scope_name + "bn4b2_branch2c")

                    self._weights.append( w_conv1_12)
                    self._weights.append( w_conv2_12)
                    self._weights.append( w_conv3_12)

                with tf.name_scope( "layer_13" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b2 = tf.add(res4b1_relu, tf.image.resize_bilinear(bn4b2_branch2c, [res4b1_relu.get_shape()[1].value, res4b1_relu.get_shape()[2].value,]), name = "res4b2")
                    res4b2_relu = tf.nn.relu( res4b2, name = "res4b2_relu")

                    w_conv1_13 = weight_variable( "W_conv1_13", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b3_branch2a = tf.nn.conv2d( res4b2_relu, w_conv1_13, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b3_branch2a")
                    bn4b3_branch2a = tf.layers.batch_normalization(res4b3_branch2a, training=self._is_trains, name = scope_name + "bn4b3_branch2a")
                    bn4b3_branch2a = tf.nn.relu( bn4b3_branch2a, name = "bn4b3_branch2a")

                    w_conv2_13 = weight_variable( "W_conv2_13", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b3_branch2b = tf.nn.atrous_conv2d( bn4b3_branch2a, w_conv2_13, rate = 2, padding = "SAME", name = "res4b3_branch2b")
                    bn4b3_branch2b = tf.layers.batch_normalization(res4b3_branch2b, training=self._is_trains, name = scope_name + "bn4b3_branch2b")
                    bn4b3_branch2b = tf.nn.relu( bn4b3_branch2b, name = "bn4b3_branch2b")

                    w_conv3_13 = weight_variable( "W_conv3_13", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b3_branch2c = tf.nn.conv2d( bn4b3_branch2b, w_conv3_13, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b3_branch2c")
                    bn4b3_branch2c = tf.layers.batch_normalization(res4b3_branch2c, training=self._is_trains, name = scope_name + "bn4b3_branch2c")

                    self._weights.append( w_conv1_13)
                    self._weights.append( w_conv2_13)
                    self._weights.append( w_conv3_13)

                with tf.name_scope( "layer_14" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b3 = tf.add(res4b2_relu, tf.image.resize_bilinear(bn4b3_branch2c, [res4b2_relu.get_shape()[1].value, res4b2_relu.get_shape()[2].value,]), name = "res4b3")
                    res4b3_relu = tf.nn.relu( res4b3, name = "res4b3_relu")

                    w_conv1_14 = weight_variable( "W_conv1_14", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b4_branch2a = tf.nn.conv2d( res4b3_relu, w_conv1_14, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b4_branch2a")
                    bn4b4_branch2a = tf.layers.batch_normalization(res4b4_branch2a, training=self._is_trains, name = scope_name + "bn4b4_branch2a")
                    bn4b4_branch2a = tf.nn.relu( bn4b4_branch2a, name = "bn4b4_branch2a")

                    w_conv2_14 = weight_variable( "W_conv2_14", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b4_branch2b = tf.nn.atrous_conv2d( bn4b4_branch2a, w_conv2_14, rate = 2, padding = "SAME", name = "res4b4_branch2b")
                    bn4b4_branch2b = tf.layers.batch_normalization(res4b4_branch2b, training=self._is_trains, name = scope_name + "bn4b4_branch2b")
                    bn4b4_branch2b = tf.nn.relu( bn4b4_branch2b, name = "bn4b4_branch2b")

                    w_conv3_14 = weight_variable( "W_conv3_14", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b4_branch2c = tf.nn.conv2d( bn4b4_branch2b, w_conv3_14, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b4_branch2c")
                    bn4b4_branch2c = tf.layers.batch_normalization(res4b4_branch2c, training=self._is_trains, name = scope_name + "bn4b4_branch2c")

                    self._weights.append( w_conv1_14)
                    self._weights.append( w_conv2_14)
                    self._weights.append( w_conv3_14)

                with tf.name_scope( "layer_15" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b4 = tf.add(res4b3_relu, tf.image.resize_bilinear(bn4b4_branch2c, [res4b3_relu.get_shape()[1].value, res4b3_relu.get_shape()[2].value,]), name = "res4b4")
                    res4b4_relu = tf.nn.relu( res4b4, name = "res4b4_relu")

                    w_conv1_15 = weight_variable( "W_conv1_15", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b5_branch2a = tf.nn.conv2d( res4b4_relu, w_conv1_15, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b5_branch2a")
                    bn4b5_branch2a = tf.layers.batch_normalization(res4b5_branch2a, training=self._is_trains, name = scope_name + "bn4b5_branch2a")
                    bn4b5_branch2a = tf.nn.relu( bn4b5_branch2a, name = "bn4b5_branch2a")

                    w_conv2_15 = weight_variable( "W_conv2_15", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b5_branch2b = tf.nn.atrous_conv2d( bn4b5_branch2a, w_conv2_15, rate = 2, padding = "SAME", name = "res4b5_branch2b")
                    bn4b5_branch2b = tf.layers.batch_normalization(res4b5_branch2b, training=self._is_trains, name = scope_name + "bn4b5_branch2b")
                    bn4b5_branch2b = tf.nn.relu( bn4b5_branch2b, name = "bn4b5_branch2b")

                    w_conv3_15 = weight_variable( "W_conv3_15", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b5_branch2c = tf.nn.conv2d( bn4b5_branch2b, w_conv3_15, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b5_branch2c")
                    bn4b5_branch2c = tf.layers.batch_normalization(res4b5_branch2c, training=self._is_trains, name = scope_name + "bn4b5_branch2c")

                    self._weights.append( w_conv1_15)
                    self._weights.append( w_conv2_15)
                    self._weights.append( w_conv3_15)

                with tf.name_scope( "layer_16" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b5 = tf.add(res4b4_relu, tf.image.resize_bilinear(bn4b5_branch2c, [res4b4_relu.get_shape()[1].value, res4b4_relu.get_shape()[2].value,]), name = "res4b5")
                    res4b5_relu = tf.nn.relu( res4b5, name = "res4b5_relu")

                    w_conv1_16 = weight_variable( "W_conv1_16", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b6_branch2a = tf.nn.conv2d( res4b5_relu, w_conv1_16, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b6_branch2a")
                    bn4b6_branch2a = tf.layers.batch_normalization(res4b6_branch2a, training=self._is_trains, name = scope_name + "bn4b6_branch2a")
                    bn4b6_branch2a = tf.nn.relu( bn4b6_branch2a, name = "bn4b6_branch2a")

                    w_conv2_16 = weight_variable( "W_conv2_16", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b6_branch2b = tf.nn.atrous_conv2d( bn4b6_branch2a, w_conv2_16, rate = 2, padding = "SAME", name = "res4b6_branch2b")
                    bn4b6_branch2b = tf.layers.batch_normalization(res4b6_branch2b, training=self._is_trains, name = scope_name + "bn4b6_branch2b")
                    bn4b6_branch2b = tf.nn.relu( bn4b6_branch2b, name = "bn4b6_branch2b")

                    w_conv3_16 = weight_variable( "W_conv3_16", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b6_branch2c = tf.nn.conv2d( bn4b6_branch2b, w_conv3_16, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b6_branch2c")
                    bn4b6_branch2c = tf.layers.batch_normalization(res4b6_branch2c, training=self._is_trains, name = scope_name + "bn4b6_branch2c")

                    self._weights.append( w_conv1_16)
                    self._weights.append( w_conv2_16)
                    self._weights.append( w_conv3_16)

                with tf.name_scope( "layer_17" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b6 = tf.add(res4b5_relu, tf.image.resize_bilinear(bn4b6_branch2c, [res4b5_relu.get_shape()[1].value, res4b5_relu.get_shape()[2].value,]), name = "res4b6")
                    res4b6_relu = tf.nn.relu( res4b6, name = "res4b6_relu")

                    w_conv1_17 = weight_variable( "W_conv1_17", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b7_branch2a = tf.nn.conv2d( res4b6_relu, w_conv1_17, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b7_branch2a")
                    bn4b7_branch2a = tf.layers.batch_normalization(res4b7_branch2a, training=self._is_trains, name = scope_name + "bn4b7_branch2a")
                    bn4b7_branch2a = tf.nn.relu( bn4b7_branch2a, name = "bn4b7_branch2a")

                    w_conv2_17 = weight_variable( "W_conv2_17", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b7_branch2b = tf.nn.atrous_conv2d( bn4b7_branch2a, w_conv2_17, rate = 2, padding = "SAME", name = "res4b7_branch2b")
                    bn4b7_branch2b = tf.layers.batch_normalization(res4b7_branch2b, training=self._is_trains, name = scope_name + "bn4b7_branch2b")
                    bn4b7_branch2b = tf.nn.relu( bn4b7_branch2b, name = "bn4b7_branch2b")

                    w_conv3_17 = weight_variable( "W_conv3_17", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b7_branch2c = tf.nn.conv2d( bn4b7_branch2b, w_conv3_17, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b7_branch2c")
                    bn4b7_branch2c = tf.layers.batch_normalization(res4b7_branch2c, training=self._is_trains, name = scope_name + "bn4b7_branch2c")

                    self._weights.append( w_conv1_17)
                    self._weights.append( w_conv2_17)
                    self._weights.append( w_conv3_17)

                with tf.name_scope( "layer_18" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b7 = tf.add(res4b6_relu, tf.image.resize_bilinear(bn4b7_branch2c, [res4b6_relu.get_shape()[1].value, res4b6_relu.get_shape()[2].value,]), name = "res4b7")
                    res4b7_relu = tf.nn.relu( res4b7, name = "res4b7_relu")

                    w_conv1_18 = weight_variable( "W_conv1_18", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b8_branch2a = tf.nn.conv2d( res4b7_relu, w_conv1_18, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b8_branch2a")
                    bn4b8_branch2a = tf.layers.batch_normalization(res4b8_branch2a, training=self._is_trains, name = scope_name + "bn4b8_branch2a")
                    bn4b8_branch2a = tf.nn.relu( bn4b8_branch2a, name = "bn4b8_branch2a")

                    w_conv2_18 = weight_variable( "W_conv2_18", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b8_branch2b = tf.nn.atrous_conv2d( bn4b8_branch2a, w_conv2_18, rate = 2, padding = "SAME", name = "res4b8_branch2b")
                    bn4b8_branch2b = tf.layers.batch_normalization(res4b8_branch2b, training=self._is_trains, name = scope_name + "bn4b8_branch2b")
                    bn4b8_branch2b = tf.nn.relu( bn4b8_branch2b, name = "bn4b8_branch2b")

                    w_conv3_18 = weight_variable( "W_conv3_18", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b8_branch2c = tf.nn.conv2d( bn4b8_branch2b, w_conv3_18, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b8_branch2c")
                    bn4b8_branch2c = tf.layers.batch_normalization(res4b8_branch2c, training=self._is_trains, name = scope_name + "bn4b8_branch2c")

                    self._weights.append( w_conv1_18)
                    self._weights.append( w_conv2_18)
                    self._weights.append( w_conv3_18)

                with tf.name_scope( "layer_19" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b8 = tf.add(res4b7_relu, tf.image.resize_bilinear(bn4b8_branch2c, [res4b7_relu.get_shape()[1].value, res4b7_relu.get_shape()[2].value,]), name = "res4b8")
                    res4b8_relu = tf.nn.relu( res4b8, name = "res4b8_relu")

                    w_conv1_19 = weight_variable( "W_conv1_19", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b9_branch2a = tf.nn.conv2d( res4b8_relu, w_conv1_19, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b9_branch2a")
                    bn4b9_branch2a = tf.layers.batch_normalization(res4b9_branch2a, training=self._is_trains, name = scope_name + "bn4b9_branch2a")
                    bn4b9_branch2a = tf.nn.relu( bn4b9_branch2a, name = "bn4b9_branch2a")

                    w_conv2_19 = weight_variable( "W_conv2_19", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b9_branch2b = tf.nn.atrous_conv2d( bn4b9_branch2a, w_conv2_19, rate = 2, padding = "SAME", name = "res4b9_branch2b")
                    bn4b9_branch2b = tf.layers.batch_normalization(res4b9_branch2b, training=self._is_trains, name = scope_name + "bn4b9_branch2b")
                    bn4b9_branch2b = tf.nn.relu( bn4b9_branch2b, name = "bn4b9_branch2b")

                    w_conv3_19 = weight_variable( "W_conv3_19", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b9_branch2c = tf.nn.conv2d( bn4b9_branch2b, w_conv3_19, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b9_branch2c")
                    bn4b9_branch2c = tf.layers.batch_normalization(res4b9_branch2c, training=self._is_trains, name = scope_name + "bn4b9_branch2c")

                    self._weights.append( w_conv1_19)
                    self._weights.append( w_conv2_19)
                    self._weights.append( w_conv3_19)

                with tf.name_scope( "layer_20" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b9 = tf.add(res4b8_relu, tf.image.resize_bilinear(bn4b9_branch2c, [res4b8_relu.get_shape()[1].value, res4b8_relu.get_shape()[2].value,]), name = "res4b9")
                    res4b9_relu = tf.nn.relu( res4b9, name = "res4b9_relu")

                    w_conv1_20 = weight_variable( "W_conv1_20", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b10_branch2a = tf.nn.conv2d( res4b9_relu, w_conv1_20, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b10_branch2a")
                    bn4b10_branch2a = tf.layers.batch_normalization(res4b10_branch2a, training=self._is_trains, name = scope_name + "bn4b10_branch2a")
                    bn4b10_branch2a = tf.nn.relu( bn4b10_branch2a, name = "bn4b10_branch2a")

                    w_conv2_20 = weight_variable( "W_conv2_20", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b10_branch2b = tf.nn.atrous_conv2d( bn4b10_branch2a, w_conv2_20, rate = 2, padding = "SAME", name = "res4b10_branch2b")
                    bn4b10_branch2b = tf.layers.batch_normalization(res4b10_branch2b, training=self._is_trains, name = scope_name + "bn4b10_branch2b")
                    bn4b10_branch2b = tf.nn.relu( bn4b10_branch2b, name = "bn4b10_branch2b")

                    w_conv3_20 = weight_variable( "W_conv3_20", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b10_branch2c = tf.nn.conv2d( bn4b10_branch2b, w_conv3_20, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b10_branch2c")
                    bn4b10_branch2c = tf.layers.batch_normalization(res4b10_branch2c, training=self._is_trains, name = scope_name + "bn4b10_branch2c")

                    self._weights.append( w_conv1_20)
                    self._weights.append( w_conv2_20)
                    self._weights.append( w_conv3_20)

                with tf.name_scope( "layer_21" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b10 = tf.add(res4b9_relu, tf.image.resize_bilinear(bn4b10_branch2c, [res4b9_relu.get_shape()[1].value, res4b9_relu.get_shape()[2].value,]), name = "res4b10")
                    res4b10_relu = tf.nn.relu( res4b10, name = "res4b10_relu")

                    w_conv1_21 = weight_variable( "W_conv1_21", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b11_branch2a = tf.nn.conv2d( res4b10_relu, w_conv1_21, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b11_branch2a")
                    bn4b11_branch2a = tf.layers.batch_normalization(res4b11_branch2a, training=self._is_trains, name = scope_name + "bn4b11_branch2a")
                    bn4b11_branch2a = tf.nn.relu( bn4b11_branch2a, name = "bn4b11_branch2a")

                    w_conv2_21 = weight_variable( "W_conv2_21", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b11_branch2b = tf.nn.atrous_conv2d( bn4b11_branch2a, w_conv2_21, rate = 2, padding = "SAME", name = "res4b11_branch2b")
                    bn4b11_branch2b = tf.layers.batch_normalization(res4b11_branch2b, training=self._is_trains, name = scope_name + "bn4b11_branch2b")
                    bn4b11_branch2b = tf.nn.relu( bn4b11_branch2b, name = "bn4b11_branch2b")

                    w_conv3_21 = weight_variable( "W_conv3_21", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b11_branch2c = tf.nn.conv2d( bn4b11_branch2b, w_conv3_21, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b11_branch2c")
                    bn4b11_branch2c = tf.layers.batch_normalization(res4b11_branch2c, training=self._is_trains, name = scope_name + "bn4b11_branch2c")

                    self._weights.append( w_conv1_21)
                    self._weights.append( w_conv2_21)
                    self._weights.append( w_conv3_21)

                with tf.name_scope( "layer_22" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b11 = tf.add(res4b10_relu, tf.image.resize_bilinear(bn4b11_branch2c, [res4b10_relu.get_shape()[1].value, res4b10_relu.get_shape()[2].value,]), name = "res4b11")
                    res4b11_relu = tf.nn.relu( res4b11, name = "res4b11_relu")

                    w_conv1_22 = weight_variable( "W_conv1_22", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b12_branch2a = tf.nn.conv2d( res4b11_relu, w_conv1_22, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b12_branch2a")
                    bn4b12_branch2a = tf.layers.batch_normalization(res4b12_branch2a, training=self._is_trains, name = scope_name + "bn4b12_branch2a")
                    bn4b12_branch2a = tf.nn.relu( bn4b12_branch2a, name = "bn4b12_branch2a")

                    w_conv2_22 = weight_variable( "W_conv2_22", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b12_branch2b = tf.nn.atrous_conv2d( bn4b12_branch2a, w_conv2_22, rate = 2, padding = "SAME", name = "res4b12_branch2b")
                    bn4b12_branch2b = tf.layers.batch_normalization(res4b12_branch2b, training=self._is_trains, name = scope_name + "bn4b12_branch2b")
                    bn4b12_branch2b = tf.nn.relu( bn4b12_branch2b, name = "bn4b12_branch2b")

                    w_conv3_22 = weight_variable( "W_conv3_22", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b12_branch2c = tf.nn.conv2d( bn4b12_branch2b, w_conv3_22, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b12_branch2c")
                    bn4b12_branch2c = tf.layers.batch_normalization(res4b12_branch2c, training=self._is_trains, name = scope_name + "bn4b12_branch2c")

                    self._weights.append( w_conv1_22)
                    self._weights.append( w_conv2_22)
                    self._weights.append( w_conv3_22)

                with tf.name_scope( "layer_23" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b12 = tf.add(res4b11_relu, tf.image.resize_bilinear(bn4b12_branch2c, [res4b11_relu.get_shape()[1].value, res4b11_relu.get_shape()[2].value,]), name = "res4b12")
                    res4b12_relu = tf.nn.relu( res4b12, name = "res4b12_relu")

                    w_conv1_23 = weight_variable( "W_conv1_23", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b13_branch2a = tf.nn.conv2d( res4b12_relu, w_conv1_23, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b13_branch2a")
                    bn4b13_branch2a = tf.layers.batch_normalization(res4b13_branch2a, training=self._is_trains, name = scope_name + "bn4b13_branch2a")
                    bn4b13_branch2a = tf.nn.relu( bn4b13_branch2a, name = "bn4b13_branch2a")

                    w_conv2_23 = weight_variable( "W_conv2_23", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b13_branch2b = tf.nn.atrous_conv2d( bn4b13_branch2a, w_conv2_23, rate = 2, padding = "SAME", name = "res4b13_branch2b")
                    bn4b13_branch2b = tf.layers.batch_normalization(res4b13_branch2b, training=self._is_trains, name = scope_name + "bn4b13_branch2b")
                    bn4b13_branch2b = tf.nn.relu( bn4b13_branch2b, name = "bn4b13_branch2b")

                    w_conv3_23 = weight_variable( "W_conv3_23", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b13_branch2c = tf.nn.conv2d( bn4b13_branch2b, w_conv3_23, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b13_branch2c")
                    bn4b13_branch2c = tf.layers.batch_normalization(res4b13_branch2c, training=self._is_trains, name = scope_name + "bn4b13_branch2c")

                    self._weights.append( w_conv1_23)
                    self._weights.append( w_conv2_23)
                    self._weights.append( w_conv3_23)

                with tf.name_scope( "layer_24" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b13 = tf.add(res4b12_relu, tf.image.resize_bilinear(bn4b13_branch2c, [res4b12_relu.get_shape()[1].value, res4b12_relu.get_shape()[2].value,]), name = "res4b13")
                    res4b13_relu = tf.nn.relu( res4b13, name = "res4b13_relu")

                    w_conv1_24 = weight_variable( "W_conv1_24", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b14_branch2a = tf.nn.conv2d( res4b13_relu, w_conv1_24, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b14_branch2a")
                    bn4b14_branch2a = tf.layers.batch_normalization(res4b14_branch2a, training=self._is_trains, name = scope_name + "bn4b14_branch2a")
                    bn4b14_branch2a = tf.nn.relu( bn4b14_branch2a, name = "bn4b14_branch2a")

                    w_conv2_24 = weight_variable( "W_conv2_24", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b14_branch2b = tf.nn.atrous_conv2d( bn4b14_branch2a, w_conv2_24, rate = 2, padding = "SAME", name = "res4b14_branch2b")
                    bn4b14_branch2b = tf.layers.batch_normalization(res4b14_branch2b, training=self._is_trains, name = scope_name + "bn4b14_branch2b")
                    bn4b14_branch2b = tf.nn.relu( bn4b14_branch2b, name = "bn4b14_branch2b")

                    w_conv3_24 = weight_variable( "W_conv3_24", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b14_branch2c = tf.nn.conv2d( bn4b14_branch2b, w_conv3_24, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b14_branch2c")
                    bn4b14_branch2c = tf.layers.batch_normalization(res4b14_branch2c, training=self._is_trains, name = scope_name + "bn4b14_branch2c")

                    self._weights.append( w_conv1_24)
                    self._weights.append( w_conv2_24)
                    self._weights.append( w_conv3_24)

                with tf.name_scope( "layer_25" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b14 = tf.add(res4b13_relu, tf.image.resize_bilinear(bn4b14_branch2c, [res4b13_relu.get_shape()[1].value, res4b13_relu.get_shape()[2].value,]), name = "res4b14")
                    res4b14_relu = tf.nn.relu( res4b14, name = "res4b14_relu")

                    w_conv1_25 = weight_variable( "W_conv1_25", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b15_branch2a = tf.nn.conv2d( res4b14_relu, w_conv1_25, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b15_branch2a")
                    bn4b15_branch2a = tf.layers.batch_normalization(res4b15_branch2a, training=self._is_trains, name = scope_name + "bn4b15_branch2a")
                    bn4b15_branch2a = tf.nn.relu( bn4b15_branch2a, name = "bn4b15_branch2a")

                    w_conv2_25 = weight_variable( "W_conv2_25", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b15_branch2b = tf.nn.atrous_conv2d( bn4b15_branch2a, w_conv2_25, rate = 2, padding = "SAME", name = "res4b15_branch2b")
                    bn4b15_branch2b = tf.layers.batch_normalization(res4b15_branch2b, training=self._is_trains, name = scope_name + "bn4b15_branch2b")
                    bn4b15_branch2b = tf.nn.relu( bn4b15_branch2b, name = "bn4b15_branch2b")

                    w_conv3_25 = weight_variable( "W_conv3_25", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b15_branch2c = tf.nn.conv2d( bn4b15_branch2b, w_conv3_25, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b15_branch2c")
                    bn4b15_branch2c = tf.layers.batch_normalization(res4b15_branch2c, training=self._is_trains, name = scope_name + "bn4b15_branch2c")

                    self._weights.append( w_conv1_25)
                    self._weights.append( w_conv2_25)
                    self._weights.append( w_conv3_25)

                with tf.name_scope( "layer_26" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b15 = tf.add(res4b14_relu, tf.image.resize_bilinear(bn4b15_branch2c, [res4b14_relu.get_shape()[1].value, res4b14_relu.get_shape()[2].value,]), name = "res4b15")
                    res4b15_relu = tf.nn.relu( res4b15, name = "res4b15_relu")

                    w_conv1_26 = weight_variable( "W_conv1_26", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b16_branch2a = tf.nn.conv2d( res4b15_relu, w_conv1_26, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b16_branch2a")
                    bn4b16_branch2a = tf.layers.batch_normalization(res4b16_branch2a, training=self._is_trains, name = scope_name + "bn4b16_branch2a")
                    bn4b16_branch2a = tf.nn.relu( bn4b16_branch2a, name = "bn4b16_branch2a")

                    w_conv2_26 = weight_variable( "W_conv2_26", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b16_branch2b = tf.nn.atrous_conv2d( bn4b16_branch2a, w_conv2_26, rate = 2, padding = "SAME", name = "res4b16_branch2b")
                    bn4b16_branch2b = tf.layers.batch_normalization(res4b16_branch2b, training=self._is_trains, name = scope_name + "bn4b16_branch2b")
                    bn4b16_branch2b = tf.nn.relu( bn4b16_branch2b, name = "bn4b16_branch2b")

                    w_conv3_26 = weight_variable( "W_conv3_26", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b16_branch2c = tf.nn.conv2d( bn4b16_branch2b, w_conv3_26, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b16_branch2c")
                    bn4b16_branch2c = tf.layers.batch_normalization(res4b16_branch2c, training=self._is_trains, name = scope_name + "bn4b16_branch2c")

                    self._weights.append( w_conv1_26)
                    self._weights.append( w_conv2_26)
                    self._weights.append( w_conv3_26)

                with tf.name_scope( "layer_27" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b16 = tf.add(res4b15_relu, tf.image.resize_bilinear(bn4b16_branch2c, [res4b15_relu.get_shape()[1].value, res4b15_relu.get_shape()[2].value,]), name = "res4b16")
                    res4b16_relu = tf.nn.relu( res4b16, name = "res4b16_relu")

                    w_conv1_27 = weight_variable( "W_conv1_27", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b17_branch2a = tf.nn.conv2d( res4b16_relu, w_conv1_27, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b17_branch2a")
                    bn4b17_branch2a = tf.layers.batch_normalization(res4b17_branch2a, training=self._is_trains, name = scope_name + "bn4b17_branch2a")
                    bn4b17_branch2a = tf.nn.relu( bn4b17_branch2a, name = "bn4b17_branch2a")

                    w_conv2_27 = weight_variable( "W_conv2_27", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b17_branch2b = tf.nn.atrous_conv2d( bn4b17_branch2a, w_conv2_27, rate = 2, padding = "SAME", name = "res4b17_branch2b")
                    bn4b17_branch2b = tf.layers.batch_normalization(res4b17_branch2b, training=self._is_trains, name = scope_name + "bn4b17_branch2b")
                    bn4b17_branch2b = tf.nn.relu( bn4b17_branch2b, name = "bn4b17_branch2b")

                    w_conv3_27 = weight_variable( "W_conv3_27", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b17_branch2c = tf.nn.conv2d( bn4b17_branch2b, w_conv3_27, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b17_branch2c")
                    bn4b17_branch2c = tf.layers.batch_normalization(res4b17_branch2c, training=self._is_trains, name = scope_name + "bn4b17_branch2c")

                    self._weights.append( w_conv1_27)
                    self._weights.append( w_conv2_27)
                    self._weights.append( w_conv3_27)

                with tf.name_scope( "layer_28" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b17 = tf.add(res4b16_relu, tf.image.resize_bilinear(bn4b17_branch2c, [res4b16_relu.get_shape()[1].value, res4b16_relu.get_shape()[2].value,]), name = "res4b17")
                    res4b17_relu = tf.nn.relu( res4b17, name = "res4b17_relu")

                    w_conv1_28 = weight_variable( "W_conv1_28", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b18_branch2a = tf.nn.conv2d( res4b17_relu, w_conv1_28, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b18_branch2a")
                    bn4b18_branch2a = tf.layers.batch_normalization(res4b18_branch2a, training=self._is_trains, name = scope_name + "bn4b18_branch2a")
                    bn4b18_branch2a = tf.nn.relu( bn4b18_branch2a, name = "bn4b18_branch2a")

                    w_conv2_28 = weight_variable( "W_conv2_28", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b18_branch2b = tf.nn.atrous_conv2d( bn4b18_branch2a, w_conv2_28, rate = 2, padding = "SAME", name = "res4b18_branch2b")
                    bn4b18_branch2b = tf.layers.batch_normalization(res4b18_branch2b, training=self._is_trains, name = scope_name + "bn4b18_branch2b")
                    bn4b18_branch2b = tf.nn.relu( bn4b18_branch2b, name = "bn4b18_branch2b")

                    w_conv3_28 = weight_variable( "W_conv3_28", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b18_branch2c = tf.nn.conv2d( bn4b18_branch2b, w_conv3_28, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b18_branch2c")
                    bn4b18_branch2c = tf.layers.batch_normalization(res4b18_branch2c, training=self._is_trains, name = scope_name + "bn4b18_branch2c")

                    self._weights.append( w_conv1_28)
                    self._weights.append( w_conv2_28)
                    self._weights.append( w_conv3_28)

                with tf.name_scope( "layer_29" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b18 = tf.add(res4b17_relu, tf.image.resize_bilinear(bn4b18_branch2c, [res4b17_relu.get_shape()[1].value, res4b17_relu.get_shape()[2].value,]), name = "res4b18")
                    res4b18_relu = tf.nn.relu( res4b18, name = "res4b18_relu")

                    w_conv1_29 = weight_variable( "W_conv1_29", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b19_branch2a = tf.nn.conv2d( res4b18_relu, w_conv1_29, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b19_branch2a")
                    bn4b19_branch2a = tf.layers.batch_normalization(res4b19_branch2a, training=self._is_trains, name = scope_name + "bn4b19_branch2a")
                    bn4b19_branch2a = tf.nn.relu( bn4b19_branch2a, name = "bn4b19_branch2a")

                    w_conv2_29 = weight_variable( "W_conv2_29", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b19_branch2b = tf.nn.atrous_conv2d( bn4b19_branch2a, w_conv2_29, rate = 2, padding = "SAME", name = "res4b19_branch2b")
                    bn4b19_branch2b = tf.layers.batch_normalization(res4b19_branch2b, training=self._is_trains, name = scope_name + "bn4b19_branch2b")
                    bn4b19_branch2b = tf.nn.relu( bn4b19_branch2b, name = "bn4b19_branch2b")

                    w_conv3_29 = weight_variable( "W_conv3_29", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b19_branch2c = tf.nn.conv2d( bn4b19_branch2b, w_conv3_29, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b19_branch2c")
                    bn4b19_branch2c = tf.layers.batch_normalization(res4b19_branch2c, training=self._is_trains, name = scope_name + "bn4b19_branch2c")

                    self._weights.append( w_conv1_29)
                    self._weights.append( w_conv2_29)
                    self._weights.append( w_conv3_29)

                with tf.name_scope( "layer_30" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b19 = tf.add(res4b18_relu, tf.image.resize_bilinear(bn4b19_branch2c, [res4b18_relu.get_shape()[1].value, res4b18_relu.get_shape()[2].value,]), name = "res4b19")
                    res4b19_relu = tf.nn.relu( res4b19, name = "res4b19_relu")

                    w_conv1_30 = weight_variable( "W_conv1_30", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b20_branch2a = tf.nn.conv2d( res4b19_relu, w_conv1_30, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b20_branch2a")
                    bn4b20_branch2a = tf.layers.batch_normalization(res4b20_branch2a, training=self._is_trains, name = scope_name + "bn4b20_branch2a")
                    bn4b20_branch2a = tf.nn.relu( bn4b20_branch2a, name = "bn4b20_branch2a")

                    w_conv2_30 = weight_variable( "W_conv2_30", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b20_branch2b = tf.nn.atrous_conv2d( bn4b20_branch2a, w_conv2_30, rate = 2, padding = "SAME", name = "res4b20_branch2b")
                    bn4b20_branch2b = tf.layers.batch_normalization(res4b20_branch2b, training=self._is_trains, name = scope_name + "bn4b20_branch2b")
                    bn4b20_branch2b = tf.nn.relu( bn4b20_branch2b, name = "bn4b20_branch2b")

                    w_conv3_30 = weight_variable( "W_conv3_30", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b20_branch2c = tf.nn.conv2d( bn4b20_branch2b, w_conv3_30, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b20_branch2c")
                    bn4b20_branch2c = tf.layers.batch_normalization(res4b20_branch2c, training=self._is_trains, name = scope_name + "bn4b20_branch2c")

                    self._weights.append( w_conv1_30)
                    self._weights.append( w_conv2_30)
                    self._weights.append( w_conv3_30)

                with tf.name_scope( "layer_31" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b20 = tf.add(res4b19_relu, tf.image.resize_bilinear(bn4b20_branch2c, [res4b19_relu.get_shape()[1].value, res4b19_relu.get_shape()[2].value,]), name = "res4b20")
                    res4b20_relu = tf.nn.relu( res4b20, name = "res4b20_relu")

                    w_conv1_31 = weight_variable( "W_conv1_31", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b21_branch2a = tf.nn.conv2d( res4b20_relu, w_conv1_31, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b21_branch2a")
                    bn4b21_branch2a = tf.layers.batch_normalization(res4b21_branch2a, training=self._is_trains, name = scope_name + "bn4b21_branch2a")
                    bn4b21_branch2a = tf.nn.relu( bn4b21_branch2a, name = "bn4b21_branch2a")

                    w_conv2_31 = weight_variable( "W_conv2_31", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b21_branch2b = tf.nn.atrous_conv2d( bn4b21_branch2a, w_conv2_31, rate = 2, padding = "SAME", name = "res4b21_branch2b")
                    bn4b21_branch2b = tf.layers.batch_normalization(res4b21_branch2b, training=self._is_trains, name = scope_name + "bn4b21_branch2b")
                    bn4b21_branch2b = tf.nn.relu( bn4b21_branch2b, name = "bn4b21_branch2b")

                    w_conv3_31 = weight_variable( "W_conv3_31", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b21_branch2c = tf.nn.conv2d( bn4b21_branch2b, w_conv3_31, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b21_branch2c")
                    bn4b21_branch2c = tf.layers.batch_normalization(res4b21_branch2c, training=self._is_trains, name = scope_name + "bn4b21_branch2c")

                    self._weights.append( w_conv1_31)
                    self._weights.append( w_conv2_31)
                    self._weights.append( w_conv3_31)

                with tf.name_scope( "layer_32" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b21 = tf.add(res4b20_relu, tf.image.resize_bilinear(bn4b21_branch2c, [res4b20_relu.get_shape()[1].value, res4b20_relu.get_shape()[2].value,]), name = "res4b21")
                    res4b21_relu = tf.nn.relu( res4b21, name = "res4b21_relu")

                    w_conv1_32 = weight_variable( "W_conv1_32", shape = [ 1, 1, 1024, 256], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res4b22_branch2a = tf.nn.conv2d( res4b21_relu, w_conv1_32, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b22_branch2a")
                    bn4b22_branch2a = tf.layers.batch_normalization(res4b22_branch2a, training=self._is_trains, name = scope_name + "bn4b22_branch2a")
                    bn4b22_branch2a = tf.nn.relu( bn4b22_branch2a, name = "bn4b22_branch2a")

                    w_conv2_32 = weight_variable( "W_conv2_32", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)))
                    res4b22_branch2b = tf.nn.atrous_conv2d( bn4b22_branch2a, w_conv2_32, rate = 2, padding = "SAME", name = "res4b22_branch2b")
                    bn4b22_branch2b = tf.layers.batch_normalization(res4b22_branch2b, training=self._is_trains, name = scope_name + "bn4b22_branch2b")
                    bn4b22_branch2b = tf.nn.relu( bn4b22_branch2b, name = "bn4b22_branch2b")

                    w_conv3_32 = weight_variable( "W_conv3_32", shape = [ 1, 1, 256, 1024], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 256)))
                    res4b22_branch2c = tf.nn.conv2d( bn4b22_branch2b, w_conv3_32, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res4b22_branch2c")
                    bn4b22_branch2c = tf.layers.batch_normalization(res4b22_branch2c, training=self._is_trains, name = scope_name + "bn4b22_branch2c")

                    self._weights.append( w_conv1_32)
                    self._weights.append( w_conv2_32)
                    self._weights.append( w_conv3_32)

                with tf.name_scope( "layer_33" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res4b22 = tf.add(res4b21_relu, tf.image.resize_bilinear(bn4b22_branch2c, [res4b21_relu.get_shape()[1].value, res4b21_relu.get_shape()[2].value,]), name = "res4b22")
                    res4b22_relu = tf.nn.relu( res4b22, name = "res4b22_relu")

                    w_conv1_33 = weight_variable( "W_conv1_33", shape = [ 1, 1, 1024, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res5a_branch1 = tf.nn.conv2d( res4b22_relu, w_conv1_33, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5a_branch1")
                    bn5a_branch1 = tf.layers.batch_normalization(res5a_branch1, training=self._is_trains, name = scope_name + "bn5a_branch1")

                    self._weights.append( w_conv1_33)

                with tf.name_scope( "layer_34" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    w_conv1_34 = weight_variable( "W_conv1_34", shape = [ 1, 1, 1024, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 1024)))
                    res5a_branch2a = tf.nn.conv2d( res4b22_relu, w_conv1_34, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5a_branch2a")
                    bn5a_branch2a = tf.layers.batch_normalization(res5a_branch2a, training=self._is_trains, name = scope_name + "bn5a_branch2a")
                    bn5a_branch2a = tf.nn.relu( bn5a_branch2a, name = "bn5a_branch2a")

                    w_conv2_34 = weight_variable( "W_conv2_34", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)))
                    res5a_branch2b = tf.nn.atrous_conv2d( bn5a_branch2a, w_conv2_34, rate = 4, padding = "SAME", name = "res5a_branch2b")
                    bn5a_branch2b = tf.layers.batch_normalization(res5a_branch2b, training=self._is_trains, name = scope_name + "bn5a_branch2b")
                    bn5a_branch2b = tf.nn.relu( bn5a_branch2b, name = "bn5a_branch2b")

                    w_conv3_34 = weight_variable( "W_conv3_34", shape = [ 1, 1, 512, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res5a_branch2c = tf.nn.conv2d( bn5a_branch2b, w_conv3_34, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5a_branch2c")
                    bn5a_branch2c = tf.layers.batch_normalization(res5a_branch2c, training=self._is_trains, name = scope_name + "bn5a_branch2c")

                    self._weights.append( w_conv1_34)
                    self._weights.append( w_conv2_34)
                    self._weights.append( w_conv3_34)

                with tf.name_scope( "layer_35" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res5a = tf.add(bn5a_branch1, tf.image.resize_bilinear(bn5a_branch2c, [bn5a_branch1.get_shape()[1].value, bn5a_branch1.get_shape()[2].value,]), name = "res5a")
                    res5a_relu = tf.nn.relu( res5a, name = "res5a_relu")

                    w_conv1_35 = weight_variable( "W_conv1_35", shape = [ 1, 1, 2048, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 2048)))
                    res5b_branch2a = tf.nn.conv2d( res5a_relu, w_conv1_35, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5b_branch2a")
                    bn5b_branch2a = tf.layers.batch_normalization(res5b_branch2a, training=self._is_trains, name = scope_name + "bn5b_branch2a")
                    bn5b_branch2a = tf.nn.relu( bn5b_branch2a, name = "bn5b_branch2a")

                    w_conv2_35 = weight_variable( "W_conv2_35", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)))
                    res5b_branch2b = tf.nn.atrous_conv2d( bn5b_branch2a, w_conv2_35, rate = 4, padding = "SAME", name = "res5b_branch2b")
                    bn5b_branch2b = tf.layers.batch_normalization(res5b_branch2b, training=self._is_trains, name = scope_name + "bn5b_branch2b")
                    bn5b_branch2b = tf.nn.relu( bn5b_branch2b, name = "bn5b_branch2b")

                    w_conv3_35 = weight_variable( "W_conv3_35", shape = [ 1, 1, 512, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res5b_branch2c = tf.nn.conv2d( bn5b_branch2b, w_conv3_35, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5b_branch2c")
                    bn5b_branch2c = tf.layers.batch_normalization(res5b_branch2c, training=self._is_trains, name = scope_name + "bn5b_branch2c")

                    self._weights.append( w_conv1_35)
                    self._weights.append( w_conv2_35)
                    self._weights.append( w_conv3_35)

                with tf.name_scope( "layer_36" ):
                    scope_name = tf.contrib.framework.get_name_scope()
                    res5b = tf.add(res5a_relu, tf.image.resize_bilinear(bn5b_branch2c, [res5a_relu.get_shape()[1].value, res5a_relu.get_shape()[2].value,]), name = "res5b")
                    res5b_relu = tf.nn.relu( res5b, name = "res5b_relu")

                    w_conv1_36 = weight_variable( "W_conv1_36", shape = [ 1, 1, 2048, 512], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 2048)))
                    res5c_branch2a = tf.nn.conv2d( res5b_relu, w_conv1_36, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5c_branch2a")
                    bn5c_branch2a = tf.layers.batch_normalization(res5c_branch2a, training=self._is_trains, name = scope_name + "bn5c_branch2a")
                    bn5c_branch2a = tf.nn.relu( bn5c_branch2a, name = "bn5c_branch2a")

                    w_conv2_36 = weight_variable( "W_conv2_36", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)))
                    res5c_branch2b = tf.nn.atrous_conv2d( bn5c_branch2a, w_conv2_36, rate = 4, padding = "SAME", name = "res5c_branch2b")
                    bn5c_branch2b = tf.layers.batch_normalization(res5c_branch2b, training=self._is_trains, name = scope_name + "bn5c_branch2b")
                    bn5c_branch2b = tf.nn.relu( bn5c_branch2b, name = "bn5c_branch2b")

                    w_conv3_36 = weight_variable( "W_conv3_36", shape = [ 1, 1, 512, 2048], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 512)))
                    res5c_branch2c = tf.nn.conv2d( bn5c_branch2b, w_conv3_36, strides = [ 1, 1, 1, 1], padding = "VALID", name = "res5c_branch2c")
                    bn5c_branch2c = tf.layers.batch_normalization(res5c_branch2c, training=self._is_trains, name = scope_name + "bn5c_branch2c")

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

                with tf.name_scope( "layer_final_rs_add" ):
                    fc1_voc12_c0_rs = tf.image.resize_bilinear(fc1_voc12_c0, [self._output_HW[0], self._output_HW[1]], name = "fc_voc12_c0_rs")
                    fc1_voc12_c1_rs = tf.image.resize_bilinear(fc1_voc12_c1, [self._output_HW[0], self._output_HW[1]], name = "fc_voc12_c1_rs")
                    fc1_voc12_c2_rs = tf.image.resize_bilinear(fc1_voc12_c2, [self._output_HW[0], self._output_HW[1]], name = "fc_voc12_c2_rs")
                    fc1_voc12_c3_rs = tf.image.resize_bilinear(fc1_voc12_c3, [self._output_HW[0], self._output_HW[1]], name = "fc_voc12_c3_rs")

                    fc1_voc12_rs = fc1_voc12_c0_rs + fc1_voc12_c1_rs + fc1_voc12_c2_rs + fc1_voc12_c3_rs

                    fc1_voc12_rs_add = tf.image.resize_bilinear(fc1_voc12_rs, [self._output_HW[0], self._output_HW[1]])

                with tf.name_scope( "layer_final_add_rs" ):
                    fc1_voc12_add = fc1_voc12_c0 + fc1_voc12_c1 + fc1_voc12_c2 + fc1_voc12_c3
                    fc1_voc12_add_rs = tf.image.resize_bilinear(fc1_voc12_add, [self._output_HW[0], self._output_HW[1]], name = "fc1_voc12_add_rs")

                with tf.name_scope( "layer_final"):
                    scope_name = tf.contrib.framework.get_name_scope()
                    fc1_voc12 = fc1_voc12_add_rs + fc1_voc12_rs_add

                    bn_fc1_voc12 = tf.layers.batch_normalization(fc1_voc12, training=self._is_trains, name = scope_name + "bn_fc1_voc12")
                    h_fc2 =tf.nn.relu(bn_fc1_voc12, name = "h_fc2")


        self._logits = h_fc2
        
        self._predictor = self.pixel_wise_softmax_2( self._logits)
        self._premodel_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "deeplab_v2")
        self._saver_premodel = tf.train.Saver(self._premodel_variables)

        # graph for cGAN
        
        # input for cGAN
        with tf.name_scope( "refine_GAN"):
            with tf.name_scope( "input_refiner"):
                self._inputs = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_HW[ 0], self._input_HW[ 1], self._num_class], name = "inputs")
                self._targets = tf.placeholder( dtype = tf.float32, shape = [ None, self._output_HW[ 0], self._output_HW[ 1], self._num_class], name = "targets")
                self._keep_probs = tf.placeholder( dtype = tf.float32, name = "keep_probs")
                self._is_train = tf.placeholder( dtype = tf.bool, shape = (), name = "is_train")

            #self._inputs = tf.identity(self._predictor)
            with tf.variable_scope("generator"):
                w_gen = weight_variable( "w_gen", shape = [ 1, 1, self._x.get_shape()[3].value, self._num_class], stddev = np.math.sqrt( 2.0 / ( ( self._x.get_shape()[3].value))))
                conv_gen = conv2d( self._x, w_gen, stride = 1, name = "conv_gen")
                gen_inputs = tf.add(self._inputs, conv_gen, name = "gen_inputs")
                self._weights_gan.append(w_gen)

                self._gen_out, self._weights_gan, self._biases_gan = build_generator(gen_inputs, self._is_train, self._weights_gan, self._biases_gan, self._keep_probs, gen_inputs.get_shape()[3].value, self._output_HW)
                #self._gen_out = self.pixel_wise_softmax_2( self._gen_out)

            with tf.name_scope("real_discriminator"):
                with tf.variable_scope("discriminator"):
                    self._disc_pred_4real, self._disc_logit_4real, self._weights_gan = build_discriminator(self._x, self._targets, self._weights_gan, self._is_train)

            with tf.name_scope("fake_discriminator"):
                with tf.variable_scope("discriminator", reuse = True):
                    self._disc_pred_4fake, self._disc_logit_4fake, self._weights_gan = build_discriminator(self._x, self._gen_out, self._weights_gan, self._is_train)
        
        self._saver = tf.train.Saver(max_to_keep = None)

    def pixel_wise_softmax_2( self, output_map):
        tensor_max = tf.tile( tf.reduce_max( output_map, 3, keepdims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        exponential_map = tf.exp( output_map - tensor_max)
        tensor_sum_exp = tf.tile( tf.reduce_sum( exponential_map, 3, keepdims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])

        return tf.div( exponential_map, tensor_sum_exp, name = "predictor")


    def train(self, data, output_path, epochs = 100, max_iter = 10, display_step = 1, opt_kwargs = {}):

        # get options -----
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_train_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        learning_rate = opt_kwargs.pop( "learning_rate", 0.0002)
        decay_rate = opt_kwargs.pop( "decay_rate", 0.95)
        momentum = opt_kwargs.pop( "momentum", 0.5)
        batch_size = opt_kwargs.pop( "batch_size", 1)
        seg_cost_name = opt_kwargs.pop("seg_cost_name", "L1")
        seg_weight = opt_kwargs.pop("seg_weight", 100.0)
        gan_weight = opt_kwargs.pop("gan_weight", 1.0)
        test_data = opt_kwargs.pop( "test_data", None)
        verification_path = opt_kwargs.pop( "verification_path", "verification")
        verification_batch_size = opt_kwargs.pop( "verification_batch_size", 4)
        pre_trained_model_iteration = opt_kwargs.pop( "pre_trained_model_iteration", None)
        save_model_epochs = opt_kwargs.pop( "save_model_epochs", np.arange( epochs))
        func_save_conditonal_model = opt_kwargs.pop( "func_save_conditonal_model", None)
        pre_gen_model = opt_kwargs.pop( "pre_gen_model", None)

        # set logger
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
                        "\t\t\tmax_iter : {0}\n".format( max_iter),
                        "\t\t\tepochs : {0}\n".format( epochs),
                        "\t\t\tdisplay_step : {0}\n".format( display_step),
                        "\t\t\tlearning_rate : {0}\n".format( learning_rate),
                        "\t\t\tmomentum : {0}\n".format( momentum),
                        "\t\t\tbatch_size : {0}\n".format( batch_size),
                        "\t\t\tseg_cost_name : {0}\n".format( seg_cost_name),
                        "\t\t\tseg_weight : {0}\n".format( seg_weight),
                        "\t\t\tgan_weight : {0}\n".format( gan_weight),
                        "\t\t\tpre_trained_model_iteration : {0}\n".format( str( pre_trained_model_iteration) if pre_trained_model_iteration is not None else "None"),
                        "\t\t\tsave_model_epochs : {0}\n".format( save_model_epochs),
                        "\t\tpre_gen_model: {0}\n".format(pre_gen_model if pre_gen_model is not None else "None")]
        logger.info( ''.join( logging_str))

        # START TRAIN

        for weight in self._weights_gan:
            sidx = find_nth( weight.name, '/', 2) + 1
            tf.summary.histogram( name = weight.name[ sidx : -2], values = weight)
        for bias in self._biases_gan:
            sidx = find_nth( bias.name, '/', 2) + 1
            tf.summary.histogram( name = bias.name[ sidx : -2], values = bias)

        """for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)"""


        shutil.rmtree( verification_path, ignore_errors = True)
        time.sleep( 0.100)
        os.makedirs( verification_path, exist_ok = True)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            #disc_loss = tf.reduce_mean(-(tf.log(self._disc_pred_4real + (1e-12)) + tf.log(1 - self._disc_pred_4fake + (1e-12))))##
            #disc_loss = tf.reduce_mean(-(tf.log(self._disc_pred_4real) + tf.log(1 - self._disc_pred_4fake)))##
            disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self._disc_logit_4real, labels = tf.ones_like(self._disc_pred_4real)))
            disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self._disc_logit_4fake, labels = tf.zeros_like(self._disc_pred_4fake)))
            self._disc_loss = disc_loss_real + disc_loss_fake
        with tf.name_scope("generator_loss"):
            #gen_loss_root = tf.reduce_mean(-tf.log(self._disc_pred_4fake + (1e-12)))##
            #gen_loss_root = tf.reduce_mean(-tf.log(self._disc_pred_4fake))
            gen_loss_root = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self._disc_logit_4fake, labels = tf.ones_like(self._disc_pred_4fake)))
            if seg_cost_name == "L1":
                gen_loss_seg = tf.reduce_mean(tf.abs(self._targets - self._gen_out))
            elif seg_cost_name == "dice_coefficient":
                eps = 1e-5
                for nc in range( self._num_class):
                    prediction_nc = self._gen_out[ :, :, :, nc]
                
                    prediction_nc = prediction_nc * tf.reduce_max( self._targets[ :, :, :, nc], axis = [ 1, 2], keep_dims = True)

                    intersection = tf.reduce_sum( prediction_nc * self._targets[ :, :, :, nc])
                    union = tf.reduce_sum( prediction_nc * prediction_nc) + tf.reduce_sum( self._targets[ :, :, :, nc])
                
                    if "cost" in locals():
                        gen_loss_seg += -( ( 2 * intersection + eps) / ( union + eps))
                    else:
                        gen_loss_seg = -( ( 2 * intersection + eps) / ( union + eps))
                gen_loss_seg /= self._num_class

            self._gen_loss = (gen_loss_root * gan_weight) + (gen_loss_seg * seg_weight)

        #global_step = tf.train.get_or_create_global_step()
        global_step = tf.Variable( 0, name = "global_step", trainable = False)
        extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope("discriminator_train"):
            disc_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            disc_optimizer = tf.train.AdamOptimizer(learning_rate, momentum)
            self._disc_train = disc_optimizer.minimize(self._disc_loss, var_list = disc_tvars)##
            disc_lrn = disc_optimizer._lr_t
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([self._disc_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optimizer = tf.train.AdamOptimizer(learning_rate, momentum)
                self._gen_train = gen_optimizer.minimize(self._gen_loss, var_list = gen_tvars)##
                gen_lrn = gen_optimizer._lr_t

        tf.summary.scalar("disc_loss", self._disc_loss)
        tf.summary.scalar("disc_real_loss", disc_loss_real)
        tf.summary.scalar("disc_fake_loss", disc_loss_fake)
        tf.summary.scalar("gen_loss", self._gen_loss)
        tf.summary.scalar("gen_loss_GAN", gen_loss_root)
        tf.summary.scalar("gen_loss_seg", gen_loss_seg)
        tf.summary.scalar("disc_lrn", disc_lrn)
        tf.summary.scalar("gen_lrn", gen_lrn)

        tensor_gan_loss = tf.placeholder( shape = (), dtype = tf.float32)
        tensor_l1_loss = tf.placeholder( shape = (), dtype = tf.float32)
        tf.summary.scalar( "tensor_gan_loss", tensor_gan_loss)
        tf.summary.scalar( "tensor_l1_loss", tensor_l1_loss)

        self._summary_op = tf.summary.merge_all()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables() if v.name.startswith("refine_GAN")])

        with tf.Session() as sess:
            print("parameter_count_exceptfor_deeplab: ", sess.run(parameter_count))
            summary_writer = tf.summary.FileWriter(output_path, graph = sess.graph)

            logger.info( "Start optimization")

            sess.run(tf.global_variables_initializer())

            if pre_trained_model_iteration == None:
                start_iter = 0
            else:
                start_iter = self.restore( sess, output_path, pre_trained_model_iteration)

            if -1 in save_model_epochs:
                self.save( sess, output_path, 0.0, start_iter)

            # load weigts from the pre-trained DeepLab
            self.restore_premodel(sess, pre_gen_model)  

            verification_x = np.ndarray( shape = ( ( verification_batch_size,) + self._input_HW + ( self._num_channel,)), dtype = np.float32)
            verification_y = np.ndarray( shape = ( ( verification_batch_size,) + self._output_HW + ( self._num_class,)), dtype = np.float32)
            for nb in range( verification_batch_size):
                verification_x0, verification_y0 = data.get(nb)
                verification_x[ nb, :, :, :] = verification_x0
                verification_y[ nb, :, :, :] = verification_y0

            verification_input = sess.run( self._predictor, feed_dict = { self._x: verification_x, self._y: verification_y, self._is_trains: False})
           
            fetches = {"disc_loss_real": disc_loss_real,
                       "disc_loss_fake": disc_loss_fake,
                       "disc_loss": self._disc_loss,
                       "gen_loss_GAN": gen_loss_root,
                       "gen_loss_seg": gen_loss_seg,
                       "gen_loss": self._gen_loss,
                       "gen_out": self._gen_out}

            verification_pr, error_rate = self.output_minibatch_stats( sess, fetches, None, verification_x, verification_input, verification_y, True)
            data.save_prediction_img( verification_path, "_init", verification_input, verification_y, verification_pr)

            batch_img = np.ndarray( shape = ( ( batch_size,) + self._input_HW + ( self._num_channel,)), dtype = np.float32)
            batch_y = np.ndarray( shape = ( ( batch_size,) + self._output_HW + ( self._num_class,)), dtype = np.float32)
            batch_fname = []

            for epoch in range(epochs):
                total_disc_realloss = 0.0
                total_disc_fakeloss = 0.0
                total_gen_loss = 0.0
                total_gen_gan_loss = 0.0
                total_gen_l1_loss = 0.0
                for step in range( ( epoch * max_iter), ( ( epoch + 1) * max_iter)):

                    # Extract prediction result from the pre-trained DeepLab
                    #batch_img, batch_target, fname = data.next( batch_size)
                    for nb in range( batch_size):
                        x0, y0, fname = data.next()
                        batch_img[ nb, :, :, :] = x0
                        batch_y[ nb, :, :, :] = y0
                        batch_fname.append(fname)

                    batch_input = sess.run( (self._predictor), feed_dict = {self._x: batch_img, self._is_trains: False})

                    # Feed the initial segmentation result from DeepLab to refinement GAN
                    ####################################################
                    # train Discriminator
                    fetches = {"train": self._disc_train,
                               "disc_lrn": disc_lrn,
                               "disc_loss": self._disc_loss,
                               "disc_loss_real": disc_loss_real,
                               "disc_loss_fake": disc_loss_fake,}
                    results = sess.run( fetches, feed_dict = {self._x: batch_img,
                                                              self._inputs: batch_input,
                                                              self._y: batch_y,
                                                              self._targets: batch_y,
                                                              self._keep_probs: np.float(1),
                                                              self._is_train: True})
                    total_disc_realloss += results["disc_loss_real"]
                    total_disc_fakeloss += results["disc_loss_fake"]

                    # train Generator
                    fetches = {"train": self._gen_train,
                               "gen_lrn": gen_lrn,
                               "gen_loss": self._gen_loss,
                               "gen_loss_GAN": gen_loss_root,
                               "gen_loss_seg": gen_loss_seg}
                    results = sess.run( fetches, feed_dict = {self._x: batch_img,
                                                              self._inputs: batch_input,
                                                              self._y: batch_y,
                                                              self._targets: batch_y,
                                                              self._keep_probs: np.float(1),
                                                              self._is_train: True})

                    total_gen_loss += results["gen_loss"]
                    total_gen_gan_loss += results["gen_loss_GAN"]
                    total_gen_l1_loss += results["gen_loss_seg"]
                    ####################################################

                    if step % display_step == 0:
                        fetches = {"disc_loss_real": disc_loss_real,
                                   "disc_loss_fake": disc_loss_fake,
                                   "disc_loss": self._disc_loss,
                                   "gen_loss_GAN": gen_loss_root,
                                   "gen_loss_seg": gen_loss_seg,
                                   "gen_loss": self._gen_loss,
                                   "gen_out": self._gen_out}
                        self.output_minibatch_stats( sess, fetches, start_iter + step, batch_img, batch_input, batch_y)
                        
                    

                logger.info( "Epoch {:}, Average real disc loss: {:.4f}, fake disc loss: {:.4f}, gen_loss: {:.4f}, gan loss: {:.4f}, seg loss: {:.4f}".format( epoch, ( total_disc_realloss / max_iter), ( total_disc_fakeloss / max_iter), ( total_gen_loss / max_iter), ( total_gen_gan_loss / max_iter), ( total_gen_l1_loss / max_iter)))

                fetches = {"disc_loss_real": disc_loss_real,
                           "disc_loss_fake": disc_loss_fake,
                           "disc_loss": self._disc_loss,
                           "gen_loss_GAN": gen_loss_root,
                           "gen_loss_seg": gen_loss_seg,
                           "gen_loss": self._gen_loss,
                           "gen_out": self._gen_out}
                verification_pr, error_rate = self.output_minibatch_stats( sess, fetches, None, verification_x, verification_input, verification_y, True)
                data.save_prediction_img( verification_path, "epoch_%s" % epoch, verification_input, verification_y, verification_pr)

                fetches = {"disc_loss": self._disc_loss,
                           "gen_loss_GAN": gen_loss_root,
                           "gen_loss_seg": gen_loss_seg,
                           "gen_loss": self._gen_loss,
                           "gen_out": self._gen_out}

                error_rate, cm, test_gan_loss, test_l1_loss = self.generator_output_test( sess, fetches, test_data)

                sm_feed_dict = {self._x: batch_img,
                                self._inputs: batch_input,
                                self._y: batch_y,
                                self._targets: batch_y,
                                self._keep_probs: np.float(1),
                                self._is_train: False}
                sm_feed_dict[tensor_gan_loss] = test_gan_loss
                sm_feed_dict[tensor_l1_loss] = test_l1_loss
                summary_str = sess.run( self._summary_op, feed_dict = sm_feed_dict)
                summary_writer.add_summary( summary_str, epoch)
                summary_writer.flush()

                if epoch in save_model_epochs:
                    self.save( sess, output_path, error_rate, start_iter + ( epoch) * max_iter)

                if func_save_conditonal_model is not None and test_data is not None:
                    save_paths = func_save_conditonal_model( epoch, cm)

                    for save_path in save_paths:
                        save_conditional_model_path = os.path.join( output_path, save_path)
                        shutil.rmtree( save_conditional_model_path, ignore_errors = True)
                        time.sleep( 0.100)
                        os.makedirs( save_conditional_model_path, exist_ok = True)
                        self.save( sess, save_conditional_model_path, error_rate, start_iter + ( epoch) * max_iter)

            logger.info("Optimization Finished!")

            return output_path

    def get_response( self, img, model_path, iter = -1):
        
        with tf.Session() as sess:
            
            sess.run( tf.global_variables_initializer())
            self.restore( sess, model_path, iter)

            input = sess.run( self._predictor, feed_dict = {self._x: img, self._keep_prob: np.float32(1), self._is_trains: False})

            #response = sess.run( self._predictor_gen, feed_dict = { self._inputs: input, self._keep_probs : np.float32( 1), self._is_train: False})
            response = sess.run( self._gen_out, feed_dict = { self._x: img,
                                                             self._inputs: input,
                                                             self._keep_probs : np.float32( 1),
                                                             self._is_train: False})
            
            return response

    def restore_premodel(self, sess, model_path):
        self._saver_premodel.restore(sess, model_path)

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
            #logger.info("Model restored from file: %s" % restored_model_path)
        elif type( iter) == str:
            riter = int( iter.split( "-")[ -1])
            self._saver.restore( sess, iter)
            #logger.info("Model restored from file: %s" % iter)
        else:
            raise ValueError( "iter must be type of str or int")
        return riter

    def save( self, sess, model_path, error_rate, iter):
        temp = self._model_name + "_" + '%.2f' % error_rate
        save_path = self._saver.save( sess, os.path.join( model_path, temp), iter)
        return save_path

    def output_minibatch_stats(self, sess, fetches, step, batch_x, batch_input, batch_y, is_verification = False):
        
        results = sess.run(fetches, feed_dict = {self._x: batch_x,
                                                 self._inputs: batch_input,
                                                 self._y: batch_y,
                                                 self._targets: batch_y,
                                                 self._keep_probs: np.float(1),
                                                 self._is_train: False})

        error_rate = 100.0 - ( 100.0 * np.sum( np.argmax( results["gen_out"], 3) == np.argmax( batch_y, 3)) / ( results["gen_out"].shape[ 0] * results["gen_out"].shape[ 1] * results["gen_out"].shape[ 2]))

        if is_verification:
            logger.info( "Verification error= {:.2f}%, gen_loss= {:.4f}, gen_gan_loss= {:.4f}, gen_seg_loss= {:.4f}".format( error_rate, results["gen_loss"], results["gen_loss_GAN"], results["gen_loss_seg"]))
            return results["gen_out"], error_rate
        else:
            logger.info( "Iter {:}, Minibatch disc_Loss= {:.4f}, realLoss= {:.4f}, fakeLoss= {:.4f}, gen_Loss= {:.4f}, GAN_Loss= {:.4f}, Seg_Loss= {:.4f}".format( step, results["disc_loss"], results["disc_loss_real"], results["disc_loss_fake"], results["gen_loss"], results["gen_loss_GAN"], results["gen_loss_seg"]))
            return results["gen_out"]

    def generator_output_test( self, sess, fetches, data):
                
        total_pixel_error = 0.
        
        data_num_class_wo_fake = data.num_class_wo_fake
        ACCURACY = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        PRECISION = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        TP = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        TN = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        DS = np.ndarray( shape = ( data.num_examples, data_num_class_wo_fake), dtype = np.float32)
        confusion_matrix_by_class = np.zeros( shape = ( data.num_class, data.num_class), dtype = np.int32)
        for nd in range( data.num_examples):
            x0, y0 = data.get(nd)
            total_gan_loss = 0.
            total_l1_loss = 0.

            input0 = sess.run( (self._predictor), feed_dict = { self._x: x0, self._is_trains: False})

            results = sess.run( fetches, feed_dict = {self._x: x0,
                                                      self._inputs: input0,
                                                      self._targets: y0,
                                                      self._keep_probs: np.float(1),
                                                      self._is_train: False})
            
            total_gan_loss += results["gen_loss_GAN"]
            total_l1_loss += results["gen_loss_seg"]
            argmax_pr = np.argmax( results["gen_out"], 3)
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

            pixel_error = 100.0 * np.count_nonzero( argmax_pr != argmax_gt) / ( 1 * results["gen_out"].shape[ 1] * results["gen_out"].shape[ 2])
            total_pixel_error += pixel_error
           
        formatter = '[' + ( "{:6d}," * data.num_class)[ : -1] + ']'
        logging_str = [ "Test error>>\n",
                        "\t\t\tgan_loss = {:.2f}\n".format( total_gan_loss),
                        "\t\t\tseg_loss = {:.2f}\n".format( total_l1_loss),
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
        return total_pixel_error / data.num_examples, confusion_matrix_by_class, total_gan_loss, total_l1_loss