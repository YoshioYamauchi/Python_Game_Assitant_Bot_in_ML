import numpy as np
from box01 import BoundBox
# from cy_yolo2_findboxes import box_constructor
from ..cython_utils.cy_yolo2_findboxes02 import box_constructor
import cv2
import time
import matplotlib.pyplot as plt


def reshape_image(img):
    h = 416
    w = 416
    c = 3
    imsz = cv2.resize(img, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz

def findboxes(out, ho, wo, meta):
    boxes = list()
    boxes = box_constructor(meta, out)
    print(len(boxes))
    detected_boxes = []
    for box in boxes:
        tmpBox = process_box(box, ho, wo, meta)
        if tmpBox is None: continue
        detected_boxes.append(tmpBox)
    return detected_boxes

# def shape_filter(left, right, top, bot, meta):
#     w = right - left
#     h = top - bot
#     # aspect_ratio = meta['aspect_ratio']
#     aspect_ratio = 0.25
#     margin = aspect_ratio * 0.2
#     if w/h > aspect_ratio + margin:
#         return False
#     elif w/h > aspect_ratio - margin:
#         return False
#     else:
#         return True

def process_box(b, h, w, meta):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    if max_prob > meta['threshold']:
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        # if shape_filter( left, right, top, bot, meta):
        #     return (left, right, top, bot, max_indx, max_prob)
        # else :
        #     return None
        return (left, right, top, bot, max_indx, max_prob)
    return None
