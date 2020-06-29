
import tensorflow as tf
import numpy as np

class Convolutional(object):
    def __init__(self, idx, batch_norm, activation, input_features, output_features,
                size, stride, pad):
        self.layer_type = 'convolutional'
        self.idx = idx
        self.batch_norm = batch_norm
        self.input_features = input_features
        self.output_features = output_features
        self.size = size
        self.stride = stride
        self.pad = pad
        self.activation = activation # 'relu' or 'linear'
        self.weights = {'kernel':None, 'biases':None}
        self.wshapes = {'kernel':[size, size, input_features, output_features],
                        'biases':[output_features]}
        if batch_norm:
            self.weights['moving_variance'] = None
            self.weights['moving_mean'] = None
            self.weights['gamma'] = None
            self.wshapes['moving_variance'] = [output_features]
            self.wshapes['moving_mean'] = [output_features]
            self.wshapes['gamma'] = [output_features]


class Maxpool(object):
    def __init__(self, idx, size, stride):
        self.layer_type = 'maxpool'
        self.idx = idx
        self.size = size
        self.stride = stride


def layer_info(meta):
    C = meta['C']
    B = meta['B']
    layers = []
    layers.append(Convolutional(0, True, 'relu', 3, 16, 3, 1, 1))
    layers.append(Maxpool(1, 2, 2))
    layers.append(Convolutional(2, True, 'relu', 16, 32, 3, 1, 1))
    layers.append(Maxpool(3, 2, 2))
    layers.append(Convolutional(4, True, 'relu', 32, 64, 3, 1, 1))
    layers.append(Maxpool(5, 2, 2))
    layers.append(Convolutional(6, True, 'relu', 64, 128, 3, 1, 1))
    layers.append(Maxpool(7, 2, 2))
    layers.append(Convolutional(8, True, 'relu', 128, 256, 3, 1, 1))
    layers.append(Maxpool(9, 2, 2))
    layers.append(Convolutional(10, True, 'relu', 256, 512, 3, 1, 1))
    layers.append(Maxpool(11, 2, 1))
    layers.append(Convolutional(12, True, 'relu', 512, 1024, 3, 1, 1))
    layers.append(Convolutional(13, True, 'relu', 1024, 1024, 3, 1, 1))
    layers.append(Convolutional(14, False, 'linear', 1024, B*(5+C), 1, 1, 0))
    return layers



def load_weights(layers, pretrained_weights):
    loader = Weights_loader(pretrained_weights)
    # The pretrained weights must be loaded in proper order.
    order_to_read = ['biases', 'gamma', 'moving_mean', 'moving_variance', 'kernel']

    for i in range(len(layers)):
        if layers[i].layer_type == 'maxpool':
            # Only convolutional layers have pretrained weights.
            continue
        for param in order_to_read:
            if param not in layers[i].weights:
                continue
            if i == len(layers) - 1:
                # The final convolutional layer does not have pretrained
                # weights.
                # print 'jh45a', i, param # 14 biases, 14 kernel
                shape = layers[i].wshapes[param]
                args = [0., 1e-2, shape]
                weights = np.random.normal(*args)
                # weights = np.ones(shape)*0.1
                weights = weights.astype(np.float32) # numpy array
                # if param == 'kernel':
                #     print 's7a8d5', shape
                #     ksize1, ksize2, infeatures, outfeatures = shape
                #     weights = weights.reshape([outfeatures, infeatures, ksize1, ksize2])
            else:
                weights = loader.walk(layers[i], param) # numpy array
                # if param in ['gamma', 'moving_variance', 'moving_mean']:
                #     print weights.shape
                #     weights = np.flip(weights,0)
                # print 'weights lwu5ht: ', weights.shape
            # if param == 'moving_variance':
            #     print 'e4guelkrq: ', ( layers[i].idx, weights[0])
            weights = tf.constant_initializer(weights) # tf.Constant
            if param in ['kernel', 'biases']:
                name = layers[i].layer_type + '_' + str(i) + '_' + param
                # print 'asd876f', layers[i].wshapes[param]
                weights = tf.get_variable(name,
                                              shape=layers[i].wshapes[param],
                                              dtype=tf.float32,
                                              initializer=weights)
            layers[i].weights[param] = weights


class Weights_loader(object):
    def __init__(self, pretrained_weights):
        self.pretrained_weights = pretrained_weights
        self.offset = 16 # the first 16 values are not used.

    def walk(self, layer, param):
        size = np.prod(layer.wshapes[param]) # how much to read
        end_point = self.offset + 4 * size
        weights = np.memmap(self.pretrained_weights, shape=(), mode='r', offset=self.offset,
                            dtype='({})float32,'.format(size))
        # print '987sd', self.offset
        self.offset = end_point # the starting point to read next time
        if param == 'kernel':
            # The weights of kernel is 3-dimentional, so it must be reshaped
            # before it is returned.
            # print 's7a8d5', layer.wshapes['kernel']
            # weights = weights.reshape(layer.wshapes['kernel'])
            ksize1, ksize2, infeatures, outfeatures = layer.wshapes['kernel']
            weights = weights.reshape([outfeatures, infeatures, ksize1, ksize2])
            weights = weights.transpose([2, 3, 1, 0])
        return weights
