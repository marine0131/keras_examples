#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import os.path
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import time
import cv2


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = os.path.join(PROJECT_PATH, 'model/trashnet_mobilenetv2.pb') # inception v3 model
INPUT_TENSOR_NAME = 'input_1:0'  # image input tensor in inception v3
OUTPUT_TENSOR_NAME = 'dense_3/Softmax:0'  # bottleneck tensor in inceptionv3

LABEL_PATH = os.path.join(PROJECT_PATH, 'model/trashnet_lables.txt')

with open(LABEL_PATH, 'r') as f:
    LABELS = json.load(f)


class TrashTF():
    def __init__(self):
        with tf.Session().as_default() as self.sess:
            # read inception-v3
            print("load inceptionV3 model")
            with tf.gfile.FastGFile(MODEL_NAME, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            # load inception-v3 model， return image input tensor and bottleneck tensor
            self.output_tensor, self.input_tensor= tf.import_graph_def(
                graph_def, return_elements=[
                    OUTPUT_TENSOR_NAME, INPUT_TENSOR_NAME])

        print "init success"

    def detect(self, img_path):
        # read image data
        img_data = cv2.imread(img_path)
        img_data = cv2.resize(img_data, (224, 224))
        img_data = np.expand_dims(img_data, 0)
        print('imgdatashape',img_data.shape)

        all_preds = self.sess.run(self.output_tensor, 
	        {self.input_tensor: img_data})

        return all_preds

    def __del__(self):
        self.sess.close()


if __name__ == '__main__':
    img_path = sys.argv[1]
    trash_tf = TrashTF()

    result = trash_tf.detect(img_path)

    print(result)

















