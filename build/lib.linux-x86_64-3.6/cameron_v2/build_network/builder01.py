
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os

def build_network(input_ph, layers, sess, meta):
    # print meta
    temp = tf.identity(input_ph)
    for i, layer in enumerate(layers):
        if layer.layer_type == 'convolutional':
            pad = [[0, 0]] + [[layer.pad, layer.pad]]*2 + [[0, 0]]
            temp = tf.pad(temp, pad)
            temp = tf.nn.conv2d(temp, layer.weights['kernel'], padding='VALID',
                                strides=[1] + [layer.stride]*2 + [1])
            if layer.batch_norm:
                name = 'batchnorm' + '_' + str(i)
                is_training = meta['is_training']
                param_initializers = {'gamma':layer.weights['gamma'],
                                     'moving_variance':layer.weights['moving_variance'],
                                     'moving_mean':layer.weights['moving_mean']}
                args = dict({ 'center':False, 'scale':True, 'epsilon':1e-5,
                              'scope':name, 'updates_collections':None,
                              'is_training':is_training,
                              'param_initializers':param_initializers})
                temp = slim.batch_norm(temp, **args)
            temp = tf.nn.bias_add(temp, layer.weights['biases'])
            if layers[i].activation == 'relu':
                temp = tf.maximum(0.1 * temp, temp)

        if layer.layer_type == 'maxpool' :
            temp = tf.nn.max_pool(temp, padding='SAME',
                                  ksize=[1] + [layer.size]*2 + [1],
                                  strides=[1] + [layer.stride]*2 + [1])

    # return temp
    return tf.identity(temp, name="network_out")


def build_train_op(output, meta):
    _TRAINER = dict({
	'rmsprop': tf.train.RMSPropOptimizer,
	'adadelta': tf.train.AdadeltaOptimizer,
	'adagrad': tf.train.AdagradOptimizer,
	'adagradDA': tf.train.AdagradDAOptimizer,
	'momentum': tf.train.MomentumOptimizer,
	'adam': tf.train.AdamOptimizer,
	'ftrl': tf.train.FtrlOptimizer,
	'sgd': tf.train.GradientDescentOptimizer
    })
    lr_ph = tf.placeholder(tf.float32)
    placeholders, loss = build_loss_op(output, meta)
    optimizer = _TRAINER[meta['optimizer']](lr_ph)
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients)
    return lr_ph, loss, train_op, placeholders


def expit_tensor(x):
    # sigmoid
	return 1. / (1. + tf.exp(-x))

def build_loss_op(net_out, meta):
    sprob = meta['sprob']
    sconf = meta['sconf']
    snoob = meta['snoob']
    scoor = meta['scoor']
    W = meta['S']
    H = meta['S']
    B = meta['B']
    C = meta['C']
    HW = H * W # number of grid cells = 13 * 13 = 169
    anchors = meta['anchors']

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H*W, B, 4])
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

    wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:,:,:,0] * wh[:,:,:,1]
    centers = coords[:,:,:,0:2]
    floor = centers - (wh * .5)
    ceil  = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    return placeholders, loss


def build_head_detector(temp):
    temp = conv(temp, 0, True, 'relu', 1, 8*4, 3, 1, 1)
    temp = maxpool(temp, 1, 2, 2)
    temp = conv(temp, 2, True, 'relu', 8*4, 8*8*2, 3, 1, 1)
    temp = maxpool(temp, 3, 2, 2)
    temp = conv(temp, 4, True, 'relu', 8*8*2, 8*16*4, 3, 1, 1)
    temp = maxpool(temp, 5, 2, 2)
    temp = conv(temp, 6, False, 'sigmoid', 8*16*4, 1, 1, 1, 0)
    return temp


def conv(temp, idx, bn, activation, input_features, output_features,
        size, stride, pad):
    temp = tf.pad(temp, [[0, 0]] + [[pad,pad]]*2 + [[0, 0]])
    w = np.random.normal(0., 1e-2, (size, size, input_features, output_features))
    w = tf.constant_initializer(w)
    w = tf.get_variable(name='kernel_%d'%idx, shape=(size,size,input_features,output_features),
                        dtype=tf.float32, initializer=w)
    temp = tf.nn.conv2d(temp, padding='VALID', strides=[1]+[stride]*2+[1], filter=w)
    if bn:
        # name = 'batchnorm_%d'%idx
        # gamma = np.random.normal(0.,
        # param_initializers = {'gamma':layer.weights['gamma'],
        #          'moving_variance':layer.weights['moving_variance'],
        #          'moving_mean':layer.weights['moving_mean']}
        # args = dict({ 'center':False, 'scale':True, 'epsilon':1e-5,
        #               'scope':name, 'updates_collections':None,
        #               'is_training':is_training,
        #               'param_initializers':param_initializers})
        # temp = slim.batch_norm(temp, **args)
        temp = tf.layers.batch_normalization(temp, training=False)
    b = np.random.normal(0., 1e-2, (output_features))
    b = tf.constant_initializer(b)
    b = tf.get_variable(name='bias_%d'%idx, shape=(output_features), dtype=tf.float32,
                        initializer=b)
    temp = tf.nn.bias_add(temp, b)
    if activation == 'relu':
        temp = tf.maximum(0.1*temp, temp)
    if activation == 'sigmoid':
        temp = tf.sigmoid(temp)
    return temp

def maxpool(temp, idx, size, stride):
    temp = tf.nn.max_pool(temp, padding='SAME',
                  ksize=[1] + [size]*2 + [1],
                  strides=[1] + [stride]*2 + [1])
    return temp
