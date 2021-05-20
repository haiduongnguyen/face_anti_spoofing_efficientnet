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
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import EfficientNetB7, EfficientNetB4, EfficientNetB5, EfficientNetB1, EfficientNetB0
from tensorflow.keras import layers


def build_resnet50(width, height, depth, classes):
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


def build_efficient_b7(width, height, depth, classes):
    net = EfficientNetB7(include_top= False, weights='imagenet', input_tensor=None, 
                    input_shape=(width,height,depth))
    res = net.output
    res = Flatten()(res)
    res = Dropout(0.3)(res)
    # res = Dense(400, activation='relu', name='dense1')(res)
    res = Dense(100, activation='relu', name='dense2')(res)
    fc = Dense(classes, activation='softmax', name='fc_out')(res)
    model = Model(inputs=net.input, outputs=fc)
    return model


from tensorflow.keras.layers.experimental import preprocessing


def build_efficient_net_b4(IMG_SIZE, IMG_DEPTH, num_classes):
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


def unfreeze_model_some_layers(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-25:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    

def build_efficient_net_b5(IMG_SIZE, num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB5(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights or not
    model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    bn_layer = layers.BatchNormalization()
    bn_layer.training = False
    x = bn_layer(x)

    # top_dropout_rate = 0.2
    # x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes , activation="softmax", name="pred")(x)

    # Compile
    model = Model(inputs, outputs, name="EfficientNet")

    return model

def build_efficient_net_b1(IMG_SIZE, IMG_DEPTH, num_classes):
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

def build_efficient_net_b0(IMG_SIZE, IMG_DEPTH, num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

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