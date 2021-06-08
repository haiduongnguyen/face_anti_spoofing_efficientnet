"""
first write: 4/6/2021 - do an
with 3 types of read image -> we have different array, that will affect our prediction
another factor is resize function, that depends on interpolation
    Linear - Bilinear
    Area 
    Nearest
and 3 library : opencv, tf, pil all supply resize function
this script will check if 3 function is the same
and do another task: see different between 3 interpolation
"""
import os
import numpy as np
import cv2
from PIL import Image
from numpy.core.fromnumeric import resize
from numpy.lib.function_base import diff
import tensorflow as tf
import matplotlib.pyplot as plt

def plt_display(image, title):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    a.set_title(title)

def show_diff_interpolation(img_path='', new_size=100):
    img_cv = cv2.imread(img_path)
    origin_width, origin_height , _ = img_cv.shape
    print("origin image has shape: (%d , %d)" % (origin_width, origin_height))

    bilinear_resize_cv = cv2.resize(img_cv, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

    nearest_resize_cv = cv2.resize(img_cv, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

    area_resize_cv = cv2.resize(img_cv, (new_size, new_size), interpolation=cv2.INTER_AREA)

    diff_1 = np.abs(bilinear_resize_cv - nearest_resize_cv)
    plt_display(diff_1, "linear - nearest")

    diff_2 = np.abs(bilinear_resize_cv - area_resize_cv)
    plt_display(diff_2, "linear - area")

    diff_3 = np.abs(area_resize_cv - nearest_resize_cv)
    plt_display(diff_3, "area - nearest")

    # print(diff_1[0,:,0])
    # print(diff_2[0,:,0])

    # plt.show()


def show_diff_cv_tf(img_path='', new_size=10):
    for mode_resize in ['linear', 'nearest', 'area']:
        if mode_resize == 'linear':
            cv_code = cv2.INTER_LINEAR
            tf_code = 'bilinear'
        if mode_resize == 'nearest':
            cv_code = cv2.INTER_NEAREST
            tf_code = 'nearest'
        if mode_resize == 'area':
            cv_code = cv2.INTER_AREA
            tf_code = 'area'

        img = cv2.imread(img_path)

        cv_resize = cv2.resize(img, (new_size, new_size), interpolation=cv_code)
        # print(resize_cv.dtype)

        img_tf = tf.constant(img)
        tf_resize = tf.image.resize(img_tf , (new_size,new_size), method=tf_code).numpy()
        tf_resize = np.ndarray.astype(tf_resize, np.uint8)
        # print(resize_tf.dtype)

        diff_1 = np.abs(cv_resize - tf_resize)
        plt_display(diff_1, 'cv - tf resize ' + mode_resize)
    # plt.show()


def show_diff_cv_pil(img_path='', new_size=10):
    for mode_resize in ['linear', 'nearest']:
        if mode_resize == 'linear':
            cv_code = cv2.INTER_LINEAR
            pil_code = Image.BILINEAR
        if mode_resize == 'nearest':
            cv_code = cv2.INTER_NEAREST
            pil_code = Image.NEAREST
        # if mode_resize == 'area':
        #     cv_code = cv2.INTER_AREA
        #     pil_code = not have yet

        img = cv2.imread(img_path)

        cv_resize = cv2.resize(img, (new_size, new_size), interpolation=cv_code)
        # print(resize_cv.dtype)

        img_pil = Image.fromarray(cv_resize)
        pil_resize = img_pil.resize((new_size, new_size), resample=pil_code)
        pil_resize = np.array(pil_resize)
        

        diff_1 = np.abs(cv_resize - pil_resize)
        plt_display(diff_1, 'cv - pil resize ' + mode_resize)
    # plt.show()

if __name__ == '__main__':

    img_path = '/home/duong/project/pyimage_research/code/version2_change_data/survey_topic/bk.jpg'
    show_diff_interpolation(img_path, new_size = 100)
    show_diff_cv_tf(img_path, new_size=1000)
    show_diff_cv_pil(img_path, new_size=1000)



    plt.show()














