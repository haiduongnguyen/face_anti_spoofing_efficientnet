"""
This file show difference between read image by flow from directory and opencv

"""
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
from PIL import Image


def plt_display(image, title):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    a.set_title(title)

image_size = 224

validation_dir = '/home/duong/project/pyimage_research/Data/version_2/small_data_to_test/crop_data/test'
valid_datagen = ImageDataGenerator()   

validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=1,
        shuffle= False,
        class_mode='categorical',
        interpolation="bilinear")

valid_aug = ImageDataGenerator()
valid_aug_generator = valid_aug.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=1,
        shuffle= False,
        class_mode='categorical',
        interpolation="bilinear")

list_names = valid_aug_generator.filenames

number_img = 1
for i in range(number_img):
    # convert to unsigned integers for plotting
    img_flow = next(valid_aug_generator)[0]
    img_flow = img_flow.astype('uint8')
    # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
    img_flow = np.squeeze(img_flow)
    # plot raw pixel data
    # ax[i].imshow(image)
    # ax[i].axis('off')
    plt_display(img_flow, 'flow from directory')

    img_path = os.path.join(validation_dir, list_names[i])
    print(img_path)
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    plt_display(img_cv, 'opencv')


    img_pil = Image.open(img_path)
    img_pil = np.array(img_pil)
    plt_display(img_pil, 'pil')

    diff_1 = img_flow - img_cv
    plt_display(diff_1, 'flow_from_directory - opencv')

    diff_2 = img_flow - img_pil
    plt_display(diff_2, 'flow from directory - pil')

plt.show()