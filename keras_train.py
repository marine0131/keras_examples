import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# import mobilenetv2_1 as mn
from data_generator import DataGenerator

def export_pb(model, output_dir, output_name, clear_devices=True):
    sess = keras.backend.get_session()
    output_names = [out.op.name for out in model.outputs]
    graph = sess.graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_names)

    tf.train.write_graph(frozen_graph, output_dir, name=output_name, as_text=False )
    print("save pb model success: {}".format(os.path.join(output_dir, output_name)))


# params
size = 224
batch_size = 128
epochs = 2
model_path = "./model"
data_path = "./data/TrashNet"
validation_split = 0.2
classes = 102
export_pbmodel = True


'''
prepare model
'''
# get model structure and freezed weights
base_model = keras.applications.mobilenet_v2.MobileNetV2(
        input_shape = (size, size, 3), 
        alpha =1.0,
        weights="imagenet",
        include_top=False,
        classes = 1000)

# print(base_model.summary())
# keras.utils.plot_model(base_model, to_file=os.path.join(model_path, "mobilenetv2.png"))

# add fc layer and softmax layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dense(512, activation='relu')(x)
preds = keras.layers.Dense(classes, activation='softmax')(x)

# merge model
model = keras.models.Model(inputs=base_model.input, outputs=preds)

# freeze base_model
for layer in base_model.layers:
    layer.trainable = False

'''
prepare data
'''
data_gen = DataGenerator(data_path, validation_split=validation_split)
train_generator, validation_generator = data_gen.generate(batch_size, size)
# get class name dict and save to label file
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

'''
save pb model
'''
if export_pbmodel:
    output_dir = model_path
    output_name = os.path.basename(os.path.normpath(data_path))+'.pb'
    export_pb(model, output_dir, output_name, clear_devices=True)
