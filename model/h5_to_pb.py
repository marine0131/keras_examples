from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.session_bundle import exporter

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys

def export_pb(model, output_dir, output_name, clear_devices=True):
    sess = keras.backend.get_session()
    output_names = [out.op.name for out in model.outputs]
    graph = sess.graph
    keep_var_names = None
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_names,
                freeze_var_names)

    tf.io.write_graph(frozen_graph, output_dir, name=output_name, as_text=False)
    print("save pb model to: {}".format(os.path.join(output_dir, output_name)))


# args
input_model = sys.argv[1]
layers_file = None
if len(sys.argv) > 2:
    layers_file = sys.argv[2]

# params
output_path = os.path.dirname(input_model)
output_name =  os.path.basename(os.path.normpath(input_model)).split('.')[0]+'.pb'


# load model
# base_model = keras.applications.mobilenet_v2.MobileNetV2(
#         input_shape = (224, 224, 3), 
#         alpha =1.0,
#         weights="imagenet",
#         include_top=False,
#         classes = 1000)
# x = base_model.output
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
# x = keras.layers.Dense(512, activation='relu')(x)
# preds = keras.layers.Dense(11, activation='softmax')(x)
# # merge model
# model = keras.models.Model(inputs=base_model.input, outputs=preds)
# model.load_weights(input_model)

keras.backend.set_learning_phase(0) # very important
model = keras.models.load_model(input_model)

if(layers_file):
    f = open(layers_file, "w")
    # layers_name = [layer.name for layer in model.layers]
    # print(layers_name)
    print("input layer: {}".format([inp.op.name for inp in model.inputs]), file=f)
    print("output_layer: {}".format([out.op.name for out in model.outputs]), file=f)
    f.close()

# transfer to pb
export_pb(model, output_path, output_name)
