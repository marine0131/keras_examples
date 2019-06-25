import sys
import os
import cv2
import json
import tensorflow as tf
from tensorflow import keras
# import mobilenetv2_1 as MobileNetV2
import numpy as np

def read_img(img_path):
    img_data = cv2.imread(img_path)
    img_data = cv2.resize(img_data, (224, 224))
    img_data = img_data.astype(np.float32)
    img_data = np.expand_dims(img_data, 0)
    img_data = img_data / 255.0;
    return img_data

def find_key_by_value(mydict, value):
    for key, v in mydict.items():
        if v == value:
            return key

if __name__ == '__main__':
    img_path = sys.argv[1]
    label_path = "model/trashnet_lables.txt"
    model_path = "model/trashnet_mobilenetv2.h5"

    # with keras.utils.CustomObjectScope({'relu6': MobileNetV2.relu6}):
    model = keras.models.load_model(model_path)

    with open(label_path, 'r') as f:
        classes = json.load(f)
        print(classes)

    # pred = model.predict(test_generator)
    # print(pred)

    true_cls = classes[os.path.basename(os.path.normpath(img_path))]


    cls = []
    if os.path.isdir(img_path):
        for imfile in os.listdir(img_path):
            print("reading file {}".format(imfile))
            img= os.path.join(img_path, imfile)
            all_predictions = model.predict(read_img(img))
            pred = np.argmax(all_predictions[0])
            cls.append(1 if pred==true_cls else 0)
            print(find_key_by_value(classes, pred), all_predictions[0, pred])

    accuracy = float(np.sum(np.array(cls))) / len(cls)
    print("accuracy: {}".format(accuracy))
    # else:
    #     result = trash_tf.detect(img_path)
    #     print result
