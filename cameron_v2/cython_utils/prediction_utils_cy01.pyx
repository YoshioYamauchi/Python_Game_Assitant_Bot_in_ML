





import numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_dynamic_crop(float p,
                        unsigned int tlx, unsigned int tly,
                        unsigned int brx, unsigned int bry,
                        unsigned int l):
  cdef unsigned int bbox_height = <unsigned int>(0.5*(bry - tly))
  cdef unsigned int bbox_center_x = <unsigned int>(0.5*(brx + tlx))
  cdef unsigned int bbox_center_y = <unsigned int>(0.5*(bry + tly))
  cdef unsigned int crop_height = bbox_height * 2
  cdef unsigned char c1 = p > 0.1
  cdef unsigned char c2 = bbox_height < 0.3 * l
  cdef unsigned char c3 = bbox_center_y + crop_height < l
  cdef unsigned char c4 = bbox_center_y - crop_height > 0
  cdef unsigned char c5 = bbox_center_x + crop_height < l
  cdef unsigned char c6 = bbox_center_x - crop_height > 0
  cdef unsigned int crop_center_x, crop_center_y, crop_size
  cdef unsigned char find_human
  if c1 * c2 * c3 * c4 * c5 * c6 == 1:
    crop_center_x = bbox_center_x
    crop_center_y = bbox_center_y
    crop_size = crop_height
    find_human = 1
  else:
    crop_center_x = <unsigned int>(0.5*l)
    crop_center_y = <unsigned int>(0.5*l)
    crop_size = l
    find_human = 0
  return crop_center_x, crop_center_y, crop_size, find_human



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def stretch_contrast(unsigned char[:,::1] img):
  cdef unsigned int max_x = img.shape[1]
  cdef unsigned int max_y = img.shape[0]
  cdef float lb = 4.
  cdef float ub = 100.
  new_img = np.zeros((max_y, max_x), dtype=np.uint8)
  cdef unsigned char[:,::1] new_img_view = new_img
  cdef unsigned int x, y
  cdef float tmp
  for y in range(max_y):
    for x in range(max_x):
      tmp = (255-1)*(img[y,x] - lb)/(ub - lb)
      if tmp < 0:
        new_img_view[y, x] = <unsigned char>255
      elif tmp > 255:
        new_img_view[y, x] = <unsigned char>255
      else:
        new_img_view[y, x] = <unsigned char>tmp
  return new_img





# @cython.boundscheck(False)
# @cython.wraparound(False)
# def stretch_contrast_cy(unsigned char[:,::1] img):
#   '''no threads'''
#   cdef Py_ssize_t x_max = img.shape[0]
#   cdef Py_ssize_t y_max = img.shape[1]
#   result = np.zeros((x_max, y_max), dtype=np.uint8)
#   cdef unsigned char[:,::1] result_view = result
#   cdef Py_ssize_t x, y
#   cdef float tmp
#   cdef float lb = 4.0
#   cdef float ub = 100.
#   for x in range(x_max):
#     for y in range(y_max):
#       tmp = (img[x, y] - lb)*(255 - 1)/(ub - lb)
#       if tmp < 0:
#         result_view[x, y] = 255
#       elif tmp > 255:
#         result_view[x, y] = 255
#       else:
#         result_view[x, y] = <unsigned char>tmp
#   return result
