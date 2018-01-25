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
    conv_2d = tf.nn.conv2d( x, W, strides = [ 1, stride, stride, 1], padding = "VALID", name = name)
    return conv_2d

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
    num_layer = 5
    num_feature_root = 64
    
    num_class = num_channel
    input_HW = output_HW
    filter_size = 3
    pool_size = 2
    up_size = [ 0] * ( num_layer - 1)
    dn_size = [ 0] * ( num_layer - 1)

    input_HW = output_HW
    for nl in range( num_layer - 1):
        input_HW = tuple( [ wh + filter_size // 2 * 2 * 2 for wh in input_HW])
        up_size[ nl] = input_HW
        input_HW = tuple( [ wh // pool_size for wh in input_HW])
    input_HW = tuple( [ wh + filter_size // 2 * 2 * 2 for wh in input_HW])
    for nl in range( num_layer - 2, -1, -1):
        input_HW = tuple( [ wh * pool_size for wh in input_HW])
        dn_size[ nl] = input_HW
        input_HW = tuple( [ wh + filter_size // 2 * 2 * 2 for wh in input_HW])

    convs = []
    pools = []
    dw_h_convs = []
    up_h_convs = []

    with tf.name_scope("graph_generator"):
        layer_scope_id = 0
        for nl in range( num_layer):

            with tf.name_scope( "%d_down" % layer_scope_id):
                num_feature = ( 2 ** nl) * num_feature_root

                if nl == 0:
                    w1 = weight_variable( "W_conv1", shape = [ filter_size, filter_size, num_channel, num_feature], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * num_channel)))
                else:
                    w1 = weight_variable( "W_conv1", shape = [ filter_size, filter_size, num_feature // 2, num_feature], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * ( num_feature // 2))))

                b1 = bias_variable( "b_conv1", shape = [ num_feature])
                
                w2  = weight_variable( "W_conv2", shape = [ filter_size, filter_size, num_feature, num_feature], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * ( num_feature))))
                b2 = bias_variable( "b_conv2", shape = [ num_feature])

                if nl == 0:
                    conv1 = conv2d( inputs, w1, name = "conv1")
                else:
                    conv1 = conv2d( pools[ -1], w1, name = "conv1")

                bn_conv1 = tf.contrib.layers.batch_norm(conv1 + b1, is_training = is_training)
                h_conv1 = lrelu( bn_conv1, 0.2)

                conv2 = conv2d( h_conv1, w2, name = "conv2")
                bn_conv2 = tf.contrib.layers.batch_norm(conv2 + b2, is_training = is_training)
                h_conv2 = lrelu( bn_conv2, 0.2)


                if nl < num_layer - 1:
                    h_conv2_pool = max_pool( h_conv2, pool_size)
                    pools.append( h_conv2_pool)
                                                
                dw_h_convs.append( h_conv2)
                weights.append( w1)
                weights.append( w2)
                biases.append( b1)
                biases.append( b2)
                convs.append( conv1)
                convs.append( conv2)
                layer_scope_id += 1
                    
        up_input = dw_h_convs[ -1]
        for nl in range( num_layer - 2, -1, -1):

            with tf.name_scope( "%d_up" % layer_scope_id):
                num_feature = ( 2 ** ( nl + 1)) * num_feature_root
                stddev = np.sqrt( 2 / ( ( filter_size ** 2) * num_feature))

                wd = weight_variable_deconv( "W_deconv", shape = [ 4, 4, num_feature // 2, num_feature])
                bd = bias_variable( "b_deconv", shape = [ num_feature // 2])

                deconv_shape = ( -1,) + up_size[ nl] + ( num_feature // 2,)
                
                if nl == num_layer - 2:
                    h_deconv = deconv2d( up_input, wd, pool_size) + bd
                    x_shape = tf.shape( up_input)
                else:
                    h_deconv = deconv2d( up_h_convs[ -1], wd, pool_size) + bd
                    x_shape = tf.shape( up_h_convs[ -1])

                #shape_up = tf.stack( [ x_shape[ 0], x_shape[ 1] * 2, x_shape[ 2] * 2, x_shape[ 3] // 2])
                shape_up_arr = [x_shape[ 0], x_shape[ 1] * 2, x_shape[ 2] * 2, x_shape[ 3] // 2]
                #print(shape_up)
                r_dw_h_convs = tf.image.resize_bilinear(dw_h_convs[ nl], [shape_up_arr[1], shape_up_arr[2]], name = "r_dw_h_convs")
                h_deconv_concat = tf.concat([r_dw_h_convs, h_deconv], 3)

                #h_deconv_concat = crop_and_concat( dw_h_convs[ nl], h_deconv, dn_size[ nl], up_size[ nl])

                w1 = weight_variable( "W_conv1", shape = [ filter_size, filter_size, num_feature, num_feature // 2], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * ( num_feature))))
                b1 = bias_variable( "b_conv1", shape = [ num_feature // 2])
                        
                w2 = weight_variable( "W_conv2", shape = [ filter_size, filter_size, num_feature // 2, num_feature // 2], stddev = np.math.sqrt( 2.0 / ( ( filter_size ** 2) * ( num_feature // 2))))
                b2 = bias_variable( "b_conv2", shape = [ num_feature // 2])

                conv1 = conv2d_with_dropout( h_deconv_concat, w1, keep_prob, name = "conv1")
                bn_conv1 = tf.contrib.layers.batch_norm(conv1 + b1, is_training = is_training)
                h_conv1 = tf.nn.relu( bn_conv1, "h_conv1")

                conv2 = conv2d_with_dropout( h_conv1, w2, keep_prob, name = "conv2")
                bn_conv2 = tf.contrib.layers.batch_norm(conv2 + b2, is_training = is_training)
                h_conv2 = tf.nn.relu( bn_conv2, name = "h_conv2")
                    
                up_h_convs.append( h_conv2)
                weights.append( w1)
                weights.append( w2)
                biases.append( b1)
                biases.append( b2)
                convs.append( conv1)
                convs.append( conv2)
                layer_scope_id += 1

        with tf.name_scope( "%d_output" % layer_scope_id):
            stddev = np.sqrt( 2 / num_feature_root)
            wo = weight_variable( "W_conv_output", shape = [ 1, 1, num_feature_root, num_class], stddev = stddev)
            bo = bias_variable( "b_conv_output", shape = [ num_class])
            conv_output = conv2d( up_h_convs[ -1], wo, name = "conv")
            output_map = tf.add( conv_output, bo, name = "h_conv")

            output_map = tf.tanh(output_map)
            output_map = tf.image.resize_bilinear(output_map, [output_HW[0], output_HW[1]], name = "output_map")

    return output_map, weights, biases

def build_discriminator(inputs, targets, weights, is_training = False):
    num_layer = 4
    num_feature_root = 64

    input = tf.concat([inputs, targets], axis = 3)

    convs = []
    with tf.name_scope("graph_discriminator"):
        with tf.name_scope( "layer_disc_1"):
            w1 = weight_variable( "W_conv1", shape = [ 4, 4, input.get_shape()[3].value, num_feature_root], stddev = np.math.sqrt( 2.0 / ( ( 4 ** 2) * ( num_feature_root))))
            conv1 = conv2d( tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), w1, stride = 2, name = "conv1")
            h_conv1 = lrelu(conv1, 0.2)
            weights.append(w1)
            convs.append(h_conv1)

        feature_num = num_feature_root
        for layer_scope_id in range( 2, num_layer+1, 1):
            with tf.name_scope( "layer_disc_%d" % layer_scope_id):
                feature_num = feature_num * 2
                stride = 1 if layer_scope_id == num_layer else 2

                w1 = weight_variable( "W_conv1", shape = [ 4, 4, np.uint16(feature_num/2), feature_num], stddev = np.math.sqrt( 2.0 / ( ( 4 ** 2) * ( np.uint16(feature_num/2)))))
                conv1 = conv2d(tf.pad(convs[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), w1, stride = stride, name = "conv1")
                bn_conv1 = tf.contrib.layers.batch_norm(conv1, is_training = is_training)
                h_conv1 = lrelu(bn_conv1, 0.2)
                weights.append(w1)
                convs.append(h_conv1)

        with tf.name_scope("layer_disc_5"):
            input_conv = tf.pad(convs[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            w1 = weight_variable( "W_conv1", shape = [ 4, 4, input_conv.get_shape()[-1].value, 1], stddev = np.math.sqrt( 2.0 / ( ( 4 ** 2) * ( input_conv.get_shape()[-1].value))))
            conv1 = conv2d(input_conv, w1, stride = 1, name = "conv1")
            output = tf.sigmoid(conv1)
            weights.append(w1)
            convs.append(conv1)

        with tf.name_scope("layer_disc_6"):
            conv1_height = output.get_shape()[1].value
            conv1_width = output.get_shape()[1].value

            w_fc1 = weight_variable("W_fc1", shape = [conv1_height * conv1_width * 1, 1], stddev = np.math.sqrt(2.0/(conv1_height * conv1_width * 1)))

            conv1_flat = tf.reshape(output, [-1, conv1_height * conv1_width * 1])
            fc1 = tf.matmul(conv1_flat, w_fc1, name = "fc1")
            h_fc1 = tf.nn.relu(fc1, name = "h_fc1")

            weights.append(w_fc1)
            convs.append(h_fc1)

    return convs[-1], weights

class build_refine_model(object):
    _model_name = "refine_GAN"
    def __init__(self, num_channel, num_class, output_HW, is_training = False): # inputs: segmentation result from the first network, targets: final segmentation result
        self._output_HW = output_HW
        self._input_HW = output_HW
        self._num_channel = num_channel
        self._num_class = num_class
        self._weights_gan = []
        self._biases_gan = []

        # input for cGAN
        with tf.name_scope( "input_refiner"):
                self._inputs = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_HW[ 0], self._input_HW[ 1], self._num_class], name = "inputs")
                self._targets = tf.placeholder( dtype = tf.float32, shape = [ None, self._output_HW[ 0], self._output_HW[ 1], self._num_class], name = "targets")
                self._keep_probs = tf.placeholder( dtype = tf.float32, name = "keep_probs")

        # input for DeepLab
        with tf.name_scope( "input"):
                self._x = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_HW[0], self._input_HW[ 1], self._num_channel], name = "x")
                self._y = tf.placeholder( dtype = tf.float32, shape = [ None, output_HW[ 0], output_HW[ 1], self._num_class], name = "y")
                self._keep_prob = tf.placeholder( dtype = tf.float32, name = "keep_prob")

        """with tf.name_scope( "input"):
            self._x = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_HW[0], self._input_HW[ 1], self._num_channel], name = "x")
            self._inputs = tf.placeholder( dtype = tf.float32, shape = [ None, self._output_HW[ 0], self._output_HW[ 1], self._num_class], name = "inputs")
            self._targets = tf.placeholder( dtype = tf.float32, shape = [ None, self._output_HW[ 0], self._output_HW[ 1], self._num_class], name = "targets")
            self._keep_prob = tf.placeholder( dtype = tf.float32, name = "keep_prob")"""

        # graph for cGAN
        with tf.name_scope( "graph_refiner"):
            #self._inputs = tf.identity(self._predictor)
            with tf.variable_scope("generator"):
                self._gen_out, self._weights_gan, self._biases_gan = build_generator(self._inputs, is_training, self._weights_gan, self._biases_gan, self._keep_probs, self._num_class, self._output_HW)
                self._predictor_gen = self.pixel_wise_softmax_2( self._gen_out)
            with tf.name_scope("discriminator_4real"):
                with tf.variable_scope("discriminator"):
                    self._disc_pred_4real, self._weights_gan = build_discriminator(self._inputs, self._targets, self._weights_gan, is_training)
            with tf.name_scope("discriminator_4fake"):
                with tf.variable_scope("discriminator", reuse = True):
                    self._disc_pred_4fake, self._weights_gan = build_discriminator(self._inputs, self._predictor_gen, self._weights_gan, is_training)

        # graph for DeepLab
        with tf.name_scope( "deeplab_v2"):
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
                    fc1_voc12 = fc1_voc12_add_rs + fc1_voc12_rs_add

                    w_fc2 = weight_variable("W_fc2", shape = [1, 1, self._num_class, self._num_class], stddev = np.math.sqrt( 2.0 / ( self._num_class)))
                    fc2 = tf.nn.conv2d( fc1_voc12, w_fc2, strides = [ 1, 1, 1, 1], padding = "SAME", name = "fc2")
                    self._weights.append(w_fc2)


        self._logits = fc2
        self._predictor = self.pixel_wise_softmax_2( self._logits)

    def pixel_wise_softmax_2( self, output_map):
        tensor_max = tf.tile( tf.reduce_max( output_map, 3, keep_dims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        exponential_map = tf.exp( output_map - tensor_max)
        tensor_sum_exp = tf.tile( tf.reduce_sum( exponential_map, 3, keep_dims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        return tf.div( exponential_map, tensor_sum_exp, name = "predictor")


    def train(self, data, output_path, epochs = 100, max_iter = 10, display_step = 1, opt_kwargs = {}):

        # get options -----
        logging_name = opt_kwargs.pop( "logging_name", self._model_name + "_train_" + time.strftime( "%Y%m%d-%H%M%S") + ".log")
        logging_folder = opt_kwargs.pop( "logging_folder", "./logs")
        learning_rate = opt_kwargs.pop( "learning_rate", 0.0002)
        decay_rate = opt_kwargs.pop( "decay_rate", 0.95)
        momentum = opt_kwargs.pop( "momentum", 0.5)
        batch_size = opt_kwargs.pop( "batch_size", 1)
        l1_weight = opt_kwargs.pop("l1_weight", 100.0)
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

        #shutil.rmtree( verification_path, ignore_errors = True)
        #time.sleep( 0.100)
        #os.makedirs( verification_path, exist_ok = True)

        # loss for discriminator
        self.disc_loss = tf.reduce_mean(-(tf.log(self._disc_pred_4real + (1e-12)) + tf.log(1 - self._disc_pred_4fake + (1e-12))))

        # loss for generator
        self.gen_loss_root = tf.reduce_mean(-tf.log(self._disc_pred_4fake + (1e-12)))
        self.gen_loss_L1 = tf.reduce_mean(tf.abs(self._targets - self._gen_out))
        self.gen_loss = (self.gen_loss_root * gan_weight) + (self.gen_loss_L1 * l1_weight)

        
        # optimizer for discriminator
        disc_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        disc_optimizer = tf.train.AdamOptimizer(learning_rate, momentum)
        disc_grads_and_vars = disc_optimizer.compute_gradients(self.disc_loss, var_list=disc_tvars)
        disc_train = disc_optimizer.apply_gradients(disc_grads_and_vars)

        # optimizer for generator
        with tf.control_dependencies([disc_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, momentum)
            gen_grads_and_vars = gen_optimizer.compute_gradients(self.gen_loss, var_list=gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

        EMA = tf.train.ExponentialMovingAverage(decay = 0.99)
        update_losses = EMA.apply([self.disc_loss, self.gen_loss_root, self.gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        self.disc_loss = EMA.average(self.disc_loss)
        self.gen_loss_root = EMA.average(self.gen_loss_root)
        self.gen_loss_L1=EMA.average(self.gen_loss_L1)

        self.train_group = tf.group(update_losses, incr_global_step, gen_train)

        tf.summary.scalar("disc_loss", self.disc_loss)
        tf.summary.scalar("gen_loss_root", self.gen_loss_root)
        tf.summary.scalar("gen_loss_L1", self.gen_loss_L1)
        tf.summary.scalar("learning_rate", learning_rate)
        self._summary_op = tf.summary.merge_all()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        self._refiner_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "graph_refiner")
        self._saver = tf.train.Saver(self._refiner_variables)

        self._premodel_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "deeplab_v2")
        self._saver_premodel = tf.train.Saver(self._premodel_variables)

        with tf.Session() as sess:
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
            if pre_gen_model is not None:
                self.restore_premodel(sess, pre_gen_model)

            batch_img = np.ndarray( shape = ( ( batch_size,) + self._output_HW + ( self._num_channel,)), dtype = np.float32)
            batch_input = np.ndarray( shape = ( ( batch_size,) + self._output_HW + ( self._num_class,)), dtype = np.float32) # prediction result from preModel
            batch_target = np.ndarray( shape = ( ( batch_size,) + self._output_HW + ( self._num_class,)), dtype = np.float32) # target
            batch_weight = np.ndarray( shape = ( ( batch_size,) + self._output_HW), dtype = np.float32)    
            
            pad_shape0 = ( 0, 0)
            pad_shape1 = ( 0, 0)

            verification_window_rect = ( 0, 0, self._output_HW[ 0], self._output_HW[ 1])
            verification_x, verification_y, verification_weight = data.get_window_batch( pad_shape0, pad_shape1, verification_window_rect, 0, verification_batch_size)
            verification_input = sess.run( self._predictor, feed_dict = { self._x: verification_x, self._keep_prob : np.float32( 1)})
            
            verification_pr, error_rate = self.output_verification_stats( sess, verification_input, verification_y)
            data.save_prediction_img( verification_path, "_init", verification_x, verification_y, verification_pr)

            for epoch in range(epochs):
                total_disc_loss = 0.0
                total_gen_gan_loss = 0.0
                total_gen_l1_loss = 0.0
                for step in range( ( epoch * max_iter), ( ( epoch + 1) * max_iter)):
                    for nb in range(batch_size):
                        x0, y0, weight0, mask0, fname = data.next( pad_shape0, pad_shape1)
                        #batch_input[ nb, :, :, :] = x0
                        batch_img[ nb, :, :, :] = x0
                        batch_target[ nb, :, :, :] = y0
                        batch_weight[ nb, :, :] = weight0

                    # Extract prediction result from the pre-trained DeepLab
                    batch_input = sess.run( self._predictor, feed_dict = { self._x: batch_img, self._keep_prob : np.float32( 1)})

                    _, disc_cost, gen_cost_root, gen_cost_l1 = sess.run( [ self.train_group, self.disc_loss, self.gen_loss_root, self.gen_loss_L1], feed_dict = {self._inputs: batch_input,
                                                                                                self._targets: batch_target, self._keep_probs : np.float32( 1)})


                    batch_fname = []

                    if step % display_step == 0:
                        self.output_minibatch_stats( sess, start_iter + step, batch_input, batch_target)
                        
                    total_disc_loss += disc_cost
                    total_gen_gan_loss += gen_cost_root
                    total_gen_l1_loss += gen_cost_l1

                summary_str = sess.run( self._summary_op, feed_dict = { self._inputs: batch_input, self._targets: batch_target, self._keep_probs : np.float32( 1)})
                summary_writer.add_summary( summary_str, epoch)
                summary_writer.flush()

                logger.info( "Epoch {:}, Average disc loss: {:.4f}, Average gan loss: {:.4f}, Average l1 loss: {:.4f}".format( epoch, ( total_disc_loss / max_iter), ( total_gen_gan_loss / max_iter), ( total_gen_l1_loss / max_iter)))

                verification_pr, error_rate = self.output_verification_stats( sess, verification_input, verification_y)
                data.save_prediction_img( verification_path, "epoch_%s" % epoch, verification_x, verification_y, verification_pr)

                if test_data is not None:
                    error_rate, cm = self.generator_output_test( sess, total_gen_gan_loss, test_data)

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

    def output_minibatch_stats(self, sess, step, batch_x, batch_y):
        
        disc_cost, gen_cost_root, gen_cost_l1 = sess.run([ self.disc_loss, self.gen_loss_root, self.gen_loss_L1], feed_dict = {self._inputs: batch_x,
                                                                                                self._targets: batch_y, self._keep_probs : np.float32( 1)})

        logger.info( "Iter {:}, Minibatch disc_Loss= {:.4f}, Minibatch gen_GAN_Loss= {:.4f}, Minibatch gen_L1_Loss= {:.4f}%".format( step, disc_cost, gen_cost_root, gen_cost_l1))

    def output_verification_stats( self, sess, batch_x, batch_y):

        pr, gen_cost_root, gen_cost_l1 = sess.run( [ self._predictor_gen, self.gen_loss_root, self.gen_loss_L1], feed_dict = {  self._inputs: batch_x, self._targets: batch_y, self._keep_probs : np.float32( 1)})

        error_rate = 100.0 - ( 100.0 * np.sum( np.argmax( pr, 3) == np.argmax( batch_y, 3)) / ( pr.shape[ 0] * pr.shape[ 1] * pr.shape[ 2]))
        logger.info( "Verification error= {:.2f}%, gen_gan_loss= {:.4f}, gen_l1_loss= {:.4f}".format( error_rate, gen_cost_root, gen_cost_l1))
        
        return pr, error_rate

    def generator_output_test( self, sess, cost, data):

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
            total_gan_loss = 0
            total_l1_loss = 0
            acc_loss_cnt = 0

            wx = x0
            wy = y0
            wweight = weight0

            winput = sess.run( self._predictor, feed_dict = { self._x: wx, self._keep_prob : np.float32( 1)})

            pr_, gen_cost_root, gen_cost_l1 = sess.run( [ self._predictor_gen, self.gen_loss_root, self.gen_loss_L1], feed_dict = {  self._inputs: winput, self._targets: wy, self._keep_probs : np.float32( 1)})

            x = wx
            pr += pr_
            acc_pr += 1
            gt = wy
                
            total_gan_loss += gen_cost_root
            total_l1_loss += gen_cost_l1
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
                        "\t\t\tgan_loss = {:.2f}\n".format( total_gan_loss / acc_loss_cnt),
                        "\t\t\tl1_loss = {:.2f}\n".format( total_l1_loss / acc_loss_cnt),
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