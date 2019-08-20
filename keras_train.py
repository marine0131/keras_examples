import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from data_generator import DataGenerator
from utils import h52pb, plot_history


'''
params
'''
export_pbmodel = True
PLOT = True
batch_size = 32
epochs = 100
model_path = "./model"
data_path = ("/home/wsy/suncan/fluid/data/suncan")
validation_split = 0.2
classes = 4
freeze_base_model = False
lr = 0.01
BASE_MODEL = "resnet" 
# BASE_MODEL = "mobilenetv2"
# BASE_MODEL = "nasnet"
# BASE_MODEL = "inception_resnet_v2"


'''
prepare model
'''
if BASE_MODEL == "mobilenetv2":
    size = 224
    base_model = keras.applications.mobilenet_v2.MobileNetV2(
            input_shape = (size, size, 3), 
            alpha =1.0,
            weights="imagenet",
            include_top=False)

if BASE_MODEL == "nasnet":
    size = 224
    base_model = keras.applications.nasnet.NASNetMobile(
            input_shape = (size, size, 3), 
            weights="imagenet",
            include_top=False)

elif BASE_MODEL == 'inceptionv3':
    size = 299
    base_model = keras.applications.inception_v3.InceptionV3(
            input_shape = (size, size, 3), 
            weights="imagenet",
            include_top=False)

elif BASE_MODEL == 'inception_resnet_v2':
    size = 299
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            input_shape = (size, size, 3), 
            weights="imagenet",
            include_top=False)

if BASE_MODEL == "resnet":
    size = 224
    base_model = keras.applications.resnet50.ResNet50(
            input_shape = (size, size, 3), 
            weights=None,
            include_top=False)
else:
    print("model name error: {}".format(BASE_MODEL))
    exit(-1)
# print(base_model.summary())
# keras.utils.plot_model(base_model, to_file=os.path.join(model_path, "mobilenetv2.png"))

# freeze base_model
if freeze_base_model:
    for layer in base_model.layers:
        layer.trainable = False


# add fc layer and softmax layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
# x = keras.layers.Dense(512, activation='relu')(x)
preds = keras.layers.Dense(classes, activation='softmax', name="softmax")(x)

# merge model
model = keras.models.Model(inputs=base_model.input, outputs=preds)
model.summary()

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
with open(os.path.join(model_path, "labels_for_android.txt"), "w") as f:
    sorted_names = sorted(class_name.items(), key=lambda v:v[1])
    for n in sorted_names:
        f.write(n[0] + '\n')

train_batches = len(train_generator)
validation_batches= len(validation_generator)

''' 
train model 
'''
# earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
# checkpoint = keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(model_path, "checkpoint/checkpoint-{epoch:04d}.ckpt"),
#         save_weights_only = True,
#         verbose=1,
#         period = 110
#         )


model.compile(optimizer=keras.optimizers.SGD(lr=lr),
        loss="categorical_crossentropy",
        metrics=['accuracy'])

# print("leraning rate: {}".format(keras.backend.get_value(model.optimizer.)))
sess = tf.InteractiveSession()

def scheduler(epoch):
    if epoch % 1 == 0 and epoch > 0:
        # lr = keras.backend.get_value(model.optimizer.lr)
        new_lr = tf.train.cosine_decay(lr, epoch, epochs, alpha=1e-5).eval()
        keras.backend.set_value(model.optimizer.lr, new_lr)
        print("lr changed to {:.4f}".format(new_lr))
    return keras.backend.get_value(model.optimizer.lr)

reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator,
                              steps_per_epoch = train_batches,
                              validation_steps = validation_batches,
                              epochs=epochs,
                              verbose = 1,
                              callbacks=[reduce_lr]
                              )

'''
save model
'''
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_name = os.path.join(model_path, data_path.split("/")[-1]+'_'+BASE_MODEL+".h5")

model.save(model_name)
del model

'''
save pb model
'''
if export_pbmodel:
    h52pb.save_pb(model_name)

'''
plot
'''
if PLOT:
    plot_history.plot(history)
