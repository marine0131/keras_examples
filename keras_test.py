import sys
import os
import cv2
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from utils import plot as mplt


def read_img(img_path):
    # return keras.preprocessing.image.load_img(img_path)
    return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def preprocess(impath, size):
    img = keras.preprocessing.image.load_img(impath, target_size=size)
    # img_data = cv2.resize(im, size)
    img_data = keras.preprocessing.image.img_to_array(img)
    img_data = np.expand_dims(img_data, 0)
    # img_data = img_data / 255.0;
    # img_data = keras.applications.mobilenet_v2.preprocess_input(img_data)
    img_data = keras.applications.inception_v3.preprocess_input(img_data)
    return img_data

def find_key_by_value(mydict, value):
    for key, v in mydict.items():
        if v == value:
            return key

if __name__ == '__main__':
    img_path = sys.argv[1]
    plot = True
    label_path = "model/suncan_labels.txt"
    model_path = "model/suncan_inceptionv3.h5"
    im_size = (299,299)

    # with keras.utils.CustomObjectScope({'relu6': MobileNetV2.relu6}):
    model = keras.models.load_model(model_path)
    model.summary()

    with open(label_path, 'r') as f:
        classes = json.load(f)
        print(classes)

    # pred = model.predict(test_generator)
    # print(pred)

    true_cls_name = os.path.basename(os.path.normpath(img_path))
    print("true class: {}".format(true_cls_name))
    # true_cls_name = CLS_MAP[cls_name]

    cls = []
    true_cls = []
    false_pred = []
    imfiles = []
    false_images = []
    values = []

    if os.path.isdir(img_path):
        for imfile in os.listdir(img_path):
            impath = os.path.join(img_path, imfile)
            im = cv2.imread(impath)
            all_predictions = model.predict(preprocess(impath, im_size))
            pred = np.argmax(all_predictions[0])
            pred_name = find_key_by_value(classes, pred) 
            print(pred_name, all_predictions[0, pred])
            if pred_name != true_cls_name:
                cls.append(0)
                imfiles.append(imfile)
                false_images.append(im)
                true_cls.append(true_cls_name)
                false_pred.append(pred_name)
                values.append(all_predictions[0, pred])

            else:
                cls.append(1)

        print("accuracy: {}".format(float(np.sum(np.array(cls))) / len(cls)))
    else:
        all_predictions = model.predict(read_img(img))
        print(all_predictions)

    if plot:
        mplt.plot_images_labels_prediction(imfiles, false_images, true_cls, false_pred,
                values,
                s=(3,3),
                pix=(256, 256))



