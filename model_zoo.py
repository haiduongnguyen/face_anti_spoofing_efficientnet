import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Input
from keras.layers import add
from tensorflow.keras.models import Model
# from keras import backend as K
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB1, EfficientNetB0
from tensorflow.keras import layers
import tensorflow.keras as K
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import tensorflow_lattice as tfl


def build_new_efficient_net_b0(height, width, depth, num_classes):
    inputs = layers.Input(shape=(height, width, depth))
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
    # Freeze the pretrained weights or not
    # model.trainable = True
    for layer in base_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # Rebuild top
    # ver2 and ver4
    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)

    # ver3 + 4
    # base_output = base_model.output
    # x = Conv2D(512, kernel_size = (1,1), activation='relu' )(base_output)
    # top_dropout_rate = 0.3
    # x = Dropout(top_dropout_rate)(x)
    # x = Flatten()(x)
    # x = BatchNormalization()(x)
    # x = Dense(512, activation='relu')(x)
    # x = BatchNormalization()(x)
    # top_dropout_rate = 0.3
    # x = Dropout(top_dropout_rate, name="top_dropout")(x)


    outputs = Dense(num_classes , activation="softmax", name="pred")(x)
    # Compile
    model = Model(inputs, outputs, name="EfficientNet")
    print(model.summary())

    return model



def build_new_efficient_net_b1(height, width, depth, num_classes):
    inputs = layers.Input(shape=(height, width, depth))
    base_model = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet")
    # Freeze the pretrained weights or not
    # model.trainable = True
    for layer in base_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    # top_dropout_rate = 0.2
    # x = Dropout(top_dropout_rate, name="top_dropout")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes , activation="softmax", name="pred")(x)
    # Compile
    model = Model(inputs, outputs, name="EfficientNet")
    print(model.summary())

    return model


def build_new_efficient_net_b4(height, width, depth, num_classes):
    inputs = layers.Input(shape=(height, width, depth))
    base_model = EfficientNetB4(include_top=False, input_tensor=inputs, weights="imagenet")
    # Freeze the pretrained weights or not
    # model.trainable = True
    for layer in base_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    # top_dropout_rate = 0.2
    # x = Dropout(top_dropout_rate, name="top_dropout")(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes , activation="softmax", name="pred")(x)
    # Compile
    model = Model(inputs, outputs, name="EfficientNet")
    print(model.summary())

    return model


# def unfreeze_model_some_layers(model):
#     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#     for layer in model.layers[-25:]:
#         if not isinstance(layer, layers.BatchNormalization):
#             layer.trainable = True

# def build_efficient_net_b4(IMG_SIZE, IMG_DEPTH, num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH))
    model = EfficientNetB4(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights or not
    model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    # x = layers.BatchNormalization()(x)
    bn_layer = layers.BatchNormalization()
    bn_layer.training = False
    x = bn_layer(x)

    # top_dropout_rate = 0.2
    # x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes , activation="softmax", name="pred")(x)

    # Compile
    model = Model(inputs, outputs, name="EfficientNet")

    return model

# def build_dense_net121(width, height, depth, classes):
    input_shape_densenet = (width, height, depth)

    densenet_model = K.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape_densenet,
        pooling=None )
    densenet_model.trainable = True

    # for layer in densenet_model.layers:
    #     if 'conv5' in layer.name:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    # densenet_model.summary()

    # input = K.Input(shape=(32, 32, 3))
    # preprocess = K.layers.Lambda(lambda x: tf.image.resize_images(x, (224, 224)), name='lamb')(input)

    layer = densenet_model.output

    layer = K.layers.Flatten()(layer)

    layer = K.layers.BatchNormalization()(layer)

    layer = K.layers.Dense(units=256, activation='relu')(layer)

    layer = K.layers.Dropout(0.2)(layer)

    layer = K.layers.BatchNormalization()(layer)

    layer = K.layers.Dense(units=128, activation='relu')(layer)

    layer = K.layers.Dropout(0.2)(layer)

    comeon = K.layers.Dense(units=classes, activation='softmax')(layer)

    model = K.models.Model(inputs=densenet_model.input, outputs=comeon)

    model.summary()

    return model



# def build_resnet50(width, height, depth, classes):
    # net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
    # 			   input_shape=(width, height, depth))
    # res = net.output
    # res = GlobalAveragePooling2D()(res)
    # fc = Dense(classes, activation='softmax', name='fc1000')(res)
    # model = Model(inputs=net.input, outputs=fc)

    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                input_shape=(width, height, depth))
    res = net.output
    res = Flatten()(res)
    fc = Dense(classes, activation='softmax', name='fc2')(res)
    model = Model(inputs=net.input, outputs=fc)
    # global model_name 
    # model_name = 'resnet 50'
    return model


# def build_efficient_net_b1(IMG_SIZE, IMG_DEPTH, num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH))
    model = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights or not
    model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # top_dropout_rate = 0.2
    # x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes , activation="softmax", name="pred")(x)

    # Compile
    model = Model(inputs, outputs, name="EfficientNet")

    return model