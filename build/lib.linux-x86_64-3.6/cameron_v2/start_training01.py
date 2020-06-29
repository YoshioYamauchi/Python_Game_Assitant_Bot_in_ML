
import tensorflow as tf
from .utils import learningMonitor04
import numpy as np
import cv2
import os
from simple_darkflow01 import SimpleDarkflow

from moviepy.editor import VideoFileClip
from mss import mss
from PIL import Image
import time


def demonstrate_training():
    class_name = 'terrorist_csgo'
    meta = dict()
    meta['annotation_folders'] = ['/home/salmis/DataBase/OpenDataSets/CSGO/PurifiedData/Annotations',
                                    '/home/salmis/DataBase/OpenDataSets/CSGO/SquareAnnotations',
                                    '/home/salmis/DataBase/OpenDataSets/CSGO/SquareAnnotations2',
                                    '/home/salmis/DataBase/OpenDataSets/CSGO/SquareAnnotations3',
                                    '/home/salmis/DataBase/OpenDataSets/CSGO/SquareAnnotations4']
    # meta['image_folder'] = '/home/salmis/DataBase/OpenDataSets/CSGO/PurifiedData/Images'
    meta['pretrained_weights'] = '/home/salmis/DataBase/OpenDataSets/tiny-yolo-voc.weights'
    meta['ckpt_folder'] = '/home/salmis/DataBase/Data2018_1124_01/Project11/Checkpoints01'
    meta.update({'sprob':1.0, 'sconf':5.0, 'snoob':1.0, 'scoor':1.0})
    meta.update({'lr':1e-5, 'epochs':300, 'minibatch_size':35, 'optimizer':'rmsprop',
                 'start_decay':None, 'restore':None})
    sdf = SimpleDarkflow(class_name)
    sdf.train(meta)

if __name__ == '__main__':
    demonstrate_training()
