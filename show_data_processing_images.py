from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
import os, datetime
import numpy as np
import cv2
# if read image from matolotlib, what will happen
#from matplotlib.image import imread
# my packages
from keras.preprocessing.image import ImageDataGenerator
from config import *
from model_zoo import *
from eer_calculation import cal_metric
from keras.models import load_model
from tqdm import tqdm
from model_zoo import *
from PIL import Image

def show_data_processing_keras(image_size, train_data_path):
        valid_datagen = ImageDataGenerator()   

        validation_generator = valid_datagen.flow_from_directory(
                train_data_path,
                target_size=(image_size, image_size),
                batch_size=1,
                shuffle= False,
                class_mode='categorical',
                interpolation="bilinear")

        fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15,15))
        for i in range(3):
                for j in range(4):
                        # convert to unsigned integers for plotting
                        image = next(validation_generator)[0]
                        image = image.astype('uint8')

                        # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
                        image = np.squeeze(image)

                        # plot raw pixel data
                        ax[i][j].imshow(image)
                        ax[i][j].axis('off')

        ## data augmentation
        valid_aug = ImageDataGenerator( rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.5,1.5])
        valid_aug_generator = valid_aug.flow_from_directory(
                train_data_path,
                target_size=(image_size, image_size),
                batch_size=1,
                shuffle= False,
                class_mode='categorical',
                interpolation="bilinear")
        fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15,15))
        for i in range(3):
                for j in range(4):
                        # convert to unsigned integers for plotting
                        image = next(valid_aug_generator)[0]
                        image = image.astype('uint8')
                        # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
                        image = np.squeeze(image)
                        # plot raw pixel data
                        ax[i][j].imshow(image)
                        ax[i][j].axis('off')

        plt.show()



if __name__ == '__main__':
        image_size = 224
        train_data_path = '/home/duong/project/pyimage_research/Data/version_2/small_data_to_test/crop_data/test'
        
        show_data_processing_keras(image_size, train_data_path)