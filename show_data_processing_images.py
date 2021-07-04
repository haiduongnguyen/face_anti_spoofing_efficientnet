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

def no_validation():
        return ImageDataGenerator()  

def validation_1():
        return  ImageDataGenerator( rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')

def validation_2():
        return  ImageDataGenerator( rotation_range=20,
                                horizontal_flip=True,
                                fill_mode='nearest',
                                brightness_range=[0.75,1.25])       

def flow_directory(valid_datagen, train_data_path):
        return valid_datagen.flow_from_directory( train_data_path,
                                                target_size=(image_size, image_size),
                                                batch_size=1,
                                                shuffle= False,
                                                class_mode='categorical',
                                                interpolation="bilinear" )

def show_image(generator, nrows, ncols):
        fig, ax = plt.subplots(nrows, ncols, figsize=(20,20))
        for i in range(nrows):
                for j in range(ncols):
                        # convert to unsigned integers for plotting
                        image = next(generator)[0]
                        image = image.astype('uint8')
                        # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
                        image = np.squeeze(image)
                        # plot raw pixel data
                        ax[i][j].imshow(image)
                        ax[i][j].axis('off')    

def show_data_processing_keras(image_size, train_data_path, nrows = 2, ncols = 2):
        validation_no_aug = no_validation()

        validation_no_aug_generator = flow_directory(validation_no_aug, train_data_path)

        show_image(validation_no_aug_generator, nrows, ncols)

        ## data augmentation 1
        # validation_aug_datagen= validation_1()
        
        ## data augmentation 2
        validation_aug = validation_2()

        validation_aug_generator = flow_directory(validation_aug, train_data_path)

        show_image(validation_aug_generator, nrows, ncols)

        plt.show()


if __name__ == '__main__':
        image_size = 224
        train_data_path = '/home/duong/project/pyimage_research/Data/version_2/small_data_to_test/crop_data/test'
        
        show_data_processing_keras(image_size, train_data_path, 2, 2)