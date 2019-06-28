import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from data_generator import DataGenerator
from utils import h52pb


'''
params
'''
size = 299
batch_size = 128
epochs = 60
model_path = "./model"
data_path = ("/home/whj/DataSet/trashnet")
validation_split = 0.2
classes = 18
export_pbmodel = True
BASE_MODEL = "inceptionv3" 
# BASE_MODEL = "mobilenetv2"


'''
prepare model
'''
if BASE_MODEL == "mobilenetv2":
    base_model = keras.applications.mobilenet_v2.MobileNetV2(
            input_shape = (size, size, 3), 
            alpha =1.0,
            weights="imagenet",
            include_top=False)

elif BASE_MODEL == 'inceptionv3':
    base_model = keras.applications.inception_v3.InceptionV3(
            input_shape = (size, size, 3), 
            weights="imagenet",
            include_top=False)

# print(base_model.summary())
# keras.utils.plot_model(base_model, to_file=os.path.join(model_path, "mobilenetv2.png"))

# freeze base_model
for layer in base_model.layers:
    layer.trainable = False

# add fc layer and softmax layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dense(512, activation='relu')(x)
preds = keras.layers.Dense(classes, activation='softmax', name="softmax")(x)

# merge model
model = keras.models.Model(inputs=base_model.input, outputs=preds)

'''
prepare data
'''
data_gen = DataGenerator(data_path, 
                         augmentation = True,
                         validation_split=validation_split)
train_generator, validation_generator = data_gen.generate(batch_size, size)

# save class names
class_name = validation_generator.class_indices
print(class_name)
with open(os.path.join(model_path, "labels.txt"), "w") as f:
    json.dump(class_name, f)

train_batches = len(train_generator)
validation_batches= len(validation_generator)

''' 
train model 
'''
# earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_path, "checkpoint-{epoch:04d}.ckpt"),
        save_weights_only = True,
        verbose=1,
        period = 300
        )

model.compile(optimizer=tf.train.AdamOptimizer(), 
        loss="categorical_crossentropy",
        metrics=['accuracy'])

hist = model.fit_generator(
        train_generator, 
        validation_data = validation_generator,
        steps_per_epoch = train_batches,
        validation_steps = validation_batches,
        epochs=epochs,
        verbose = 2,
        callbacks=[checkpoint]
        )

'''
save model
'''
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_name = os.path.join(model_path, data_path.split("/")[-1]+".h5")

model.save(model_name)
del model

'''
save pb model
'''
if export_pbmodel:
    h52pb.save_pb(model_name)
