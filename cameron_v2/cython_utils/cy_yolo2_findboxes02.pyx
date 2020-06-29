import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from ..prediction.box01 import BoundBox
# from nms cimport NMS

#expit
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float expit_c(float x):
    cdef float y= 1/(1+exp(-x))
    return y

#MAX
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float max_c(float a, float b):
    if(a>b):
        return a
    return b


#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_constructor(meta, np.ndarray[float,ndim=3] net_out_in):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1,index,index2
        np.intp_t max_Pobj_H, max_Pobj_B, max_Pobj_W
        float max_Pobj = 0
        # float  threshold = meta['threshold']
        float tempc,arr_max=0,sum=0
        float prob_obj, relative_center_x, relative_center_y, relative_w, relative_h
        # double[:] anchors = np.asarray([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52])
        # double[:] anchors = np.asarray([0.4, 1.2, 0.6, 1.82, 0.82, 2.45, 1.16, 3.48, 1.5, 4.5])
        # double[:] anchors = np.asarray([0.4, 1.2, 0.6, 1.82, 0.75, 2.26, 0.95, 2.80, 1.12, 3.35])
        # double[:] anchors = np.asarray([0.82, 2.45])
        double [:] anchors = np.asarray([0.4, 1.2, 0.66, 1.98, 0.92, 2.76, 1.18, 3.54, 1.44, 4.32, 1.7, 5.1])
        # double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H = 13
    W = 13
    C = 1
    # B = 5
    # B = 2
    # B = 1
    B = 6

    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)

    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                Pobj = expit_c(Bbox_pred[row, col, box_loop, 4])
                if Pobj > max_Pobj:
                  max_Pobj_H = row
                  max_Pobj_W = col
                  max_Pobj_B = box_loop
                  max_Pobj = Pobj

    prob_obj = expit_c(Bbox_pred[max_Pobj_H, max_Pobj_W, max_Pobj_B, 4]) # sigmoid(P(obj))
    relative_center_x = (max_Pobj_W + expit_c(Bbox_pred[max_Pobj_H, max_Pobj_W, max_Pobj_B, 0])) / W
    relative_center_y = (max_Pobj_H + expit_c(Bbox_pred[max_Pobj_H, max_Pobj_W, max_Pobj_B, 1])) / H
    relative_w = exp(Bbox_pred[max_Pobj_H, max_Pobj_W, max_Pobj_B, 2]) * anchors[2 * max_Pobj_B + 0] / W
    relative_h = exp(Bbox_pred[max_Pobj_H, max_Pobj_W, max_Pobj_B, 3]) * anchors[2 * max_Pobj_B + 1] / H
    # return (relative_center_x, relative_center_y, relative_w, relative_h, prob_obj)
    return (relative_center_x, relative_center_y, relative_w, relative_h, prob_obj, max_Pobj_B)

    #NMS
    # return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
