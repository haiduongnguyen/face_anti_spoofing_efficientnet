from config import *
from model_zoo import *
from eer_calculation import cal_metric
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import keras
import tensorflow as tf
from keras.applications import VGG16
import os, datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load full model (.h5 file)
model_name = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet/result_20210515/training_checkpoint/efficient_net_b4/cp_02.hdf5'
model = load_model(model_name)




# Note that the validation data should not be augmented!
valid_datagen = ImageDataGenerator()   
validation_dir = crop_data_test
validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')





# keras evaluate on validation data
model.evaluate(validation_generator, batch_size=1)





