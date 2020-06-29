import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from .build_network.layer01 import layer_info, load_weights
from .build_network.builder01 import build_network, build_train_op, build_head_detector
import tensorflow as tf
from .training.trainer01 import start_training
# import utils.learningMonitor04
from .utils import learningMonitor04
import numpy as np
import cv2
from .prediction.predict02 import reshape_image, mva_findbox, findbox
import time


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())


class SimpleDarkflow(object):
    _start_training = start_training
    def __init__(self, class_name):
        self.meta = {'class_name':class_name,
                     'C':1, 'S':13}
        # self.meta['anchors'] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
        # self.meta['anchors'] = [1., 1., 3., 3., 5., 5., 8., 8.]
        # self.meta['anchors'] = [0.38, 1.8, 0.72, 3.0, 1.0, 4.3, 1.6, 6.2, 2.3, 8.5]
        # self.meta['anchors'] = [0.4, 1.2, 0.6, 1.82, 0.82, 2.45, 1.16, 3.48, 1.5, 4.5]
        # self.meta['anchors'] = [0.4, 1.2, 0.6, 1.82, 0.75, 2.26, 0.95, 2.80, 1.12, 3.35]
        # self.meta['anchors'] = [0.6, 1.82, 0.95, 2.80]
        # self.meta['anchors'] = [0.82, 2.45]
        self.meta['anchors'] = [0.4, 1.2, 0.66, 1.98, 0.92, 2.76, 1.18, 3.54, 1.44, 4.32, 1.7, 5.1]
        self.meta['B'] = int(len(self.meta['anchors'])/2)
        anchors_np = np.array(self.meta['anchors']).reshape(self.meta['B'], 2)
        self.meta['aspect_ratio'] = np.mean(anchors_np[:, 0]/anchors_np[:, 1])
        self.layers = layer_info(self.meta)
        self.placeholders = dict()
        # self.placeholders = {'input':tf.placeholder(tf.float32, [None, 416, 416, 3], 'input')}
        # self.placeholders['crop_input'] = tf.placeholder(tf.float32, [None, 128, 128, 3], 'input')
        self.prepare = True

    def train(self, meta):
        self.placeholders = {'input':tf.placeholder(tf.float32, [None, 416, 416, 3], 'input')}
        self.monitor = learningMonitor04.LearningMonitor()
        self.meta.update(meta)
        self.meta['is_training'] = True
        load_weights(self.layers, self.meta['pretrained_weights'])
        self.graph = tf.Graph()
        cfg = {#'allow_soft_placement':True,
               #'log_device_placement':False,
               'gpu_options':tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)}
        self.sess = tf.Session(config = tf.ConfigProto(**cfg))
        self.sess.run(tf.global_variables_initializer())
        with tf.device('/device:GPU:0'):
            self.output = build_network(self.placeholders['input'], self.layers, self.sess, self.meta)
        # with tf.Session() as sess:
            self.placeholders['lr'], self.loss, self.train_op, placeholders = build_train_op(self.output, self.meta)
        self.placeholders.update(placeholders)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        if self.meta['restore']:
            self.saver.restore(self.sess, os.path.join(meta['ckpt_folder'], meta['restore']))
        self._start_training()
        self.sess.close()

    def return_predict(self, meta, image):
        image = cv2.imread('grace_hopper.jpg')
        if self.prepare:
            self._prepare_predict(meta)
        ho, wo, co = image.shape
        reshaped = reshape_image(image)
        reshaped = np.expand_dims(reshaped, 0)
        feed_dict = {self.placeholders['input']:reshaped}
        fetches = [self.output]
        fetched = self.sess.run(fetches, feed_dict)
        out = fetched[0]
        print("length of output: ", out.flatten.shape[0])
        print("last element: ", out.flatten[13*13*30])
        print("first element: ", out.flatten[0])
        time.sleep(100000)
        detected_box, B = findbox(out[0], ho, wo, meta)
        tlx, tly, brx, bry, pobj = detected_box
        boxw = brx - tlx
        boxh = bry - tly
        box_xc = int(0.5*(brx + tlx))
        box_yc = int(0.5*(bry + tly))
        cimg_tly = max(int(box_yc - 0.5*boxh - 0.1*boxh), 0)
        cimg_bry = min(int(box_yc - 0.5*boxh + 0.1*boxh), ho)
        cimgh = cimg_bry - cimg_tly
        cimg_tlx = int(box_xc - 0.5*cimgh)
        cimg_brx = int(box_xc + 0.5*cimgh)
        # print cimg_tlx, cimg_tly, cimg_brx, cimg_bry
        cimg = image[cimg_tly:cimg_bry, cimg_tlx:cimg_brx, :]
        if cimg.shape[0] != 0 and cimg.shape[1] != 0:
            cimg = self.to_grayscale(cimg)
            cimg = self.strech_contrast(cimg)
            cimg = self.preprocess(cimg)
            cimg = np.expand_dims(cimg, -1)
            cimg = np.expand_dims(cimg, 0)
            cout = self.sess2.run(self.crop_out, {self.placeholders['crop_input']: cimg})
            cout = cout[0,:,:]
            rshift = np.zeros_like(cout)
            rshift[:,1:] = cout[:,:-1]
            lshift = np.zeros_like(cout)
            lshift[:,:-1] = cout[:,1:]
            tshift = np.zeros_like(cout)
            tshift[:-1,:] = cout[1:,:]
            bshift = np.zeros_like(cout)
            bshift[1:,:] = cout[:-1,:]
            cout = cout * rshift * lshift * bshift * tshift
            max_idx = np.argmax(cout.flatten())
            row = int(max_idx / 16)
            col = int(max_idx % 16)
            hx = min(int(cimgh * (col + 0.5)/16) + cimg_tlx, wo-1)
            hy = min(int(cimgh * (row + 0.5)/16) + cimg_tly, ho-1)
            # detected_box = mva_findbox(out[0], ho, wo, meta)
        else:
            hx, hy = None, None
        return detected_box, B, (hx, hy)

    def to_grayscale(self, img):
        return  img[:,:,2]

    def preprocess(self, img):
        h = 128
        w = 128
        img = cv2.resize(img, (w, h))
        img = img / 255.
        return img

    def strech_contrast(self, img):
        lb = 4.
        ub = 100.
        img = (255.-1)/(ub-lb)*(img*1. - lb)
        img[img < 0] = 255
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img

    def simple_edge_detector(self, img):
        img = img * 1.
        new_img = np.zeros_like(img)
        size = img.shape[0]
        # d = int(img.shape[0]*0.02)
        d = 4
        th = 60
        crop1 = img[d:,d:]
        crop2 = img[:-d, d:]
        crop3 = img[:-d, :-d]
        crop4 = img[d:, :-d]
        # f = np.abs(crop1 - crop3) > th
        f1 = np.abs(crop1 - crop2) > th
        f1 = f1 + (np.abs(crop1 - crop3) > th)
        f1 = f1 + (np.abs(crop1 - crop4) > th)
        # f1 = f1 + (np.abs(crop2 - crop3) > th)
        # f1 = f1 + (np.abs(crop2 - crop4) > th)
        # f1 = f1 + (np.abs(crop3 - crop4) > th)
        new_img[int(d/2):size-int(d/2), int(d/2):size-int(d/2)] = f1 * np.ones((img.shape[0]-d, img.shape[1]-d))
        # return np.abs(crop1 - crop2).astype(np.uint8)
        return new_img

    def _prepare_predict(self, meta):
        self.meta.update(meta)
        self.meta['is_training'] = False
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.placeholders['input'] = tf.placeholder(tf.float32, [None, 416, 416, 3], 'input')
            load_weights(self.layers, self.meta['pretrained_weights'])
            cfg = {#'allow_soft_placement':False,
                   #'log_device_placement':False,
                   'gpu_options':tf.GPUOptions(per_process_gpu_memory_fraction = 1),
                   # 'device_count':{'GPU':2}
                   }
            self.sess = tf.Session(config = tf.ConfigProto(**cfg))
            with tf.device('/device:GPU:0'):
                self.output = build_network(self.placeholders['input'], self.layers, self.sess, self.meta)
                # update_ops -
            # self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(tf.global_variables())
            # self.saver = tf.train.Saver()
            self.saver.restore(self.sess, os.path.join(meta['ckpt_folder'], meta['ckpt']))
            self.prepare = False





        # os.environ["CUDA_VISIBLE_DEVICES"]="1"
        hd_ckpt_folder = '/media/salmis10/PrivateData/ExternalDataBase/Data2018_1124_01/Project12/GPU1_Checkpoint'
        hd_ckpt_name = 'gpu1_ckpt'
        self.graph2 = tf.Graph()
        with self.graph2.as_default():
            self.placeholders['crop_input'] = tf.placeholder(tf.float32, [None, 128, 128, 1], 'input')
            self.sess2 = tf.Session()
            with tf.device('/device:GPU:0'):
                self.crop_out = build_head_detector(self.placeholders['crop_input'])
            # self.sess2.run(tf.global_variables_initializer())
            self.saver2 = tf.train.Saver(tf.global_variables())
            # self.saver2 = tf.train.Saver()
            self.saver2.restore(self.sess2, os.path.join(hd_ckpt_folder, hd_ckpt_name))
