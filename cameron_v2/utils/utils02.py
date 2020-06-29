from numpy.random import permutation as perm
from copy import deepcopy
import pickle
import numpy as np
import cv2
import os
import sys
import xml.etree.ElementTree as ET
import glob

def read_xml(meta):
    # folder : path to the folder that contains xml annotations
    dumps = list()
    cur_dir = os.getcwd()
    filepaths = []
    for xml_folder in meta['annotation_folders']:
        filepaths += glob.glob(xml_folder+ '/*.xml')
    size = len(filepaths)
    print(size)
    for i, filepath in enumerate(filepaths):
        in_file = open(filepath)
        # print filepath
        tree = ET.parse(in_file)
        root = tree.getroot()
        image_name = str(root.find('filename').text)
        image_folder = str(root.find('folder').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()
        for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text
                # xmlbox = obj.find('bndbox')
                xmlbox = obj.find('bbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                current = [name,xn,yn,xx,yx]
                all += [current]
        add = [[image_name, image_folder, [w, h, all]]]
        dumps += add
        in_file.close()
    return dumps

def values_loss_ph(meta, chunk):
    # chunk : a list containing image name, iamge size, and objects
    S = meta['S']
    B = meta['B']
    C = meta['C']
    W = meta['S']
    H = meta['S']
    labels = [meta['class_name']]
    # preprocess
    jpg = chunk[0];
    image_folder = chunk[1]
    w, h, allobj_ = chunk[2]
    allobj = deepcopy(allobj_)
    # print '87asdf89', allobj
    # path = os.path.join(meta['image_folder'], jpg)
    path = os.path.join(image_folder, jpg)
    img = preprocess(meta, path, allobj)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H:
            # print 'asdfk45werk: '
            return None, None
        obj[3] = float(obj[3]-obj[1]) / w
        obj[4] = float(obj[4]-obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]
    # print '567k56jklh: ', allobj

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    for obj in allobj:
        probs[obj[5], :, :] = [[0.]*C] * B
        probs[obj[5], :, labels.index(obj[0])] = 1.
        proid[obj[5], :, :] = [[1.]*C] * B
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * W # xleft
        prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * H # yup
        prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * W # xright
        prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * H # ybot
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft;
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }

    return inp_feed_val, loss_feed_val


def preprocess(meta, im, allobj = None):
    if type(im) is not np.ndarray:
        im = cv2.imread(im)
    # im = add_gaussian_noise(im)
    if allobj is not None: # in training mode
        # result = imcv2_affine_trans(im)
        result = random_crop(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip: continue
            obj_1_ =  obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        im = imcv2_recolor(im)
    im = resize_input(meta, im)
    if allobj is None: return im
    return im#, np.array(im) # for unit testing

def add_gaussian_noise(im):
    h, w, c = im.shape
    stdev = 40
    noise = stdev * np.abs(np.random.randn(h, w, c))
    add_noise = im + noise <= 255
    noised_im = im + noise * add_noise
    noised_im = noised_im.astype(np.uint8)
    return noised_im


def resize_input(meta, im):
    h = 416
    w = 416
    c = 3
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    # imsz = imsz[:,:,::-1] # is this process necessary?
    return imsz


def random_crop(im):
	h, w, c = im.shape
	scale = np.random.uniform() / 10. + 1.
	max_offx = (scale-1.) * w
	max_offy = (scale-1.) * h
	offx = int(np.random.uniform() * max_offx)
	offy = int(np.random.uniform() * max_offy)

	im = cv2.resize(im, (0,0), fx = scale, fy = scale)
	im = im[offy : (offy + h), offx : (offx + w)]
	flip = np.random.binomial(1, .5)
	if flip: im = cv2.flip(im, 1)
	return im, [w, h, c], [scale, [offx, offy], flip]

def imcv2_recolor(im, a = .1):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1. # between -1 and +1

    # random amplify each channel
    im = im * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    # up = 0.3 * 2 - 1
    im = cv2.pow(im/mx, 1. + up * .5)
    return np.array(im * 255., np.uint8)



def _fix(obj, dims, scale, offs):
	for i in range(1, 5):
		dim = dims[(i + 1) % 2]
		off = offs[(i + 1) % 2]
		obj[i] = int(obj[i] * scale - off)
		obj[i] = max(min(obj[i], dim), 0)

def generate_traininig_data(meta):
    data = read_xml(meta)
    data_size=  len(data)
    # print 'a8sd76f', data_size # 2394
    batch = meta['minibatch_size']
    batch_per_epoch = int(data_size / batch)
    # print 'o87sdf', batch_per_epoch # 149
    for i in range(meta['epochs']):
        shuffle_idx = perm(np.arange(data_size))
        for b in range(batch_per_epoch):
            # yield these
            x_batch = list()
            feed_batch = dict()
            for j in range(b*batch, b*batch+batch):
                train_instance = data[ shuffle_idx[j] ]
                try:
                    inp, new_feed = values_loss_ph(meta, train_instance)
                except ZeroDivisionError:
                    print("This image's width or height are zeros: ", train_instance[0])
                    print('train_instance:', train_instance)
                    print('Please remove or fix it then try again.')
                    raise

                if inp is None: continue
                x_batch += [np.expand_dims(inp, 0)]
                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key, np.zeros((0,) + new.shape))
                    # Because this is mini-batch training, the training data
                    # must be concatenated into a single numpy array, may be.
                    feed_batch[ key ] = np.concatenate([ old_feed, [new] ])

            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch


def get_lr(meta, step):
    if step < meta['start_decay'] or meta['start_decay'] == None :
        return meta['lr']
    else :
        x = step - meta['start_decay']
        return meta['lr'] * np.exp(-0.1*x)
