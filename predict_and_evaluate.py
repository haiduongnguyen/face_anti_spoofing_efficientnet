import numpy as np
from numpy.core.fromnumeric import argmin
import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_lattice as tfl
import os, datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import statistics
# my packages
from config import *
from model_zoo import *
from eer_calculation import cal_metric
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.models import load_model


model = load_model('/home/duong/project/pyimage_research/version2_change_data/result_ver04/cp_04.hdf5')


test_dir = '/home/duong/project/pyimage_research/version2_change_data/small_data_to_test/crop_data/test'

test_datagen = ImageDataGenerator()   
 
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        shuffle = False,
        batch_size=1,
        class_mode='categorical')

# opt_adam = keras.optimizers.Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", metrics=['binary_accuracy', 'categorical_accuracy'])
# a = model.evaluate(test_generator, batch_size=1)
# print(a)


filenames = test_generator.filenames
nb_samples = len(filenames)

prediction = model.predict(test_generator,steps = nb_samples)
print("prediction has shape : ", prediction.shape)
spoof_score = np.array(prediction)[:,1]
print(spoof_score)
# print(predict)
predict_classed = np.argmax(prediction, axis = 1)

label = np.array([0]*118 + [1]*114)
print("label has shape : ", label.shape)

temp = 0
for i in range(nb_samples):
    if predict_classed[i] == label[i]:
        temp += 1

predict_acc = temp/nb_samples
print("acc of predict is : ", predict_acc)




