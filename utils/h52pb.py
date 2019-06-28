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


def save_pb(input_model):
    # params
    output_path = os.path.dirname(input_model)
    output_name =  os.path.basename(os.path.normpath(input_model)).split('.')[0]+'.pb'

    # change model learning phase to 0
    keras.backend.set_learning_phase(0) # very important
    # load model
    model = keras.models.load_model(input_model)

    # print input output layers to file
    f = open(os.path.join(output_path, "input_output_layers.txt"), "w")
    print("input layer: {}".format([inp.op.name for inp in model.inputs]), file=f)
    print("output_layer: {}".format([out.op.name for out in model.outputs]), file=f)
    f.close()

    # transfer to pb
    export_pb(model, output_path, output_name)


if __name__ == "__main__":
    # args
    input_model = sys.argv[1]
    save_pb(input_model)

