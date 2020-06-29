import numpy as np
# from box01 import BoundBox
# from cy_yolo2_findboxes import box_constructor
from ..cython_utils.cy_yolo2_findboxes02 import box_constructor
import cv2
import time
import matplotlib.pyplot as plt
import time


def reshape_image(img):
    h = 416
    w = 416
    c = 3
    # t00 = time.time()
    imsz = cv2.resize(img, (w, h))
    imsz = imsz / 255.
    print("this is new function reshape_image()")
    imsz = imsz[:,:,::-1]
    # reordered = np.zeros(shape=(h,w,c))
    # reordered[:,:,0] = imsz[:,:,1]
    # reordered[:,:,1] = imsz[:,:,0]
    # reordered[:,:,2] = imsz[:,:,2]
    #
    # imsz = reordered
    # t01 = time.time()
    # print("j452: {:5.4f} ms".format(1000*(t01-t00)))
    return imsz


# def findboxes(out, ho, wo, meta):
#     boxes = list()
#     boxes = box_constructor(meta, out)
#     print len(boxes)
#     detected_boxes = []
#     for box in boxes:
#         tmpBox = process_box(box, ho, wo, meta)
#         if tmpBox is None: continue
#         detected_boxes.append(tmpBox)
#     return detected_boxes

def findbox(out, ho, wo, meta):
    rcx, rcy, rw, rh, pobj, B = box_constructor(meta, out)
    # print("asdkjh: ", rcx, rcy, rw, rh, pobj, B)
    print("rcx: ", rcx)
    print("rcy: ", rcy)
    print("rw: ", rw)
    print("rh: ", rh)
    print("pobj: ", pobj)
    print("B: ", B)
    tlx = int((rcx - 0.5*rw)*wo)
    tly = int((rcy - 0.5*rh)*ho)
    brx = int((rcx + 0.5*rw)*wo)
    bry = int((rcy + 0.5*rh)*ho)
    if tlx < 0: tlx = 0
    if tly < 0: tly = 0
    return (tlx, tly, brx, bry, pobj), B


rw_mva = 0.1
rh_mva = 0.1
frame_count = 1
def mva_findbox(out, ho, wo, meta):
    global rw_mva
    global rh_mva
    global frame_count
    rcx, rcy, rw, rh, pobj = box_constructor(meta, out)
    ratio = 0.6
    thr = 0.1
    epsilon = 1e-3
    if np.abs((rw - rw_mva)/(rw_mva+epsilon)) < thr:
        rw_mva = rw_mva * ratio + rw * (1 - ratio)
    elif frame_count % 5 == 0:
        rw_mva = rw_mva * ratio + rw * (1 - ratio)
    if np.abs((rh - rh_mva)/(rh_mva+epsilon)) < thr:
        rh_mva = rh_mva * ratio + rh * (1 - ratio)
    elif frame_count % 5 == 0:
        rh_mva = rh_mva * ratio + rh * (1 - ratio)
    tlx = int((rcx - 0.5*rw_mva)*wo)
    tly = int((rcy - 0.5*rh_mva)*ho)
    brx = int((rcx + 0.5*rw_mva)*wo)
    bry = int((rcy + 0.5*rh_mva)*ho)
    frame_count += 1
    return (tlx, tly, brx, bry, pobj)

def return_head_position(out):
    pass
