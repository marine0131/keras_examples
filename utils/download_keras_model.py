import tensorflow as tf
from tensorflow import keras

# base_model = keras.applications.nasnet.NASNetMobile(
#             input_shape = (224, 224, 3), 
#             weights="imagenet",
#             include_top=True)

# base_model = keras.applications.mobilenet.MobileNet(
#             input_shape = (224, 224, 3), 
#             weights="imagenet",
#             include_top=True)

base_model = keras.applications.vgg16.VGG16(
            input_shape = (224, 224, 3), 
            weights="imagenet",
            include_top=True)
keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
