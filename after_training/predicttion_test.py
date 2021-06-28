import cv2
import tensorflow as tf
import os, datetime
import numpy as np
# my packages

from keras.models import load_model


model_name = './resnet50_xin'

model = load_model(model_name)



img_path = 'KH_16085308540121620090303.jpeg'

img = cv2.imread(img_path)

image = tf.convert_to_tensor(img, 'float32')/255

image = tf.expand_dims(image, 0)

score = model.predict(image)

print(score)
