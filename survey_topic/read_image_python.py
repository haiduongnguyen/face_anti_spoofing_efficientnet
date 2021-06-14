"""
first write: 4/6/2021
read an image from a file, 3 library opencv, tensorflow, pil supply different function that lead to different array
I follow the idea in this blog: https://towardsdatascience.com/image-read-and-resize-with-opencv-tensorflow-and-pil-3e0f29b992be

"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def plt_display(image, title):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    a.set_title(title)



img_path = '/home/duong/project/pyimage_research/code/version2_change_data/survey_topic/bk.jpg'
# img_path = '/home/duong/project/pyimage_research/Data/version_2/small_data_to_test/crop_data/test/live/KH_16085067860121620065249.jpg'

# read by open cv
img_cv = cv2.imread(img_path)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
plt_display(img_cv, 'origin image')

# read by tensorflow
img_tf = tf.io.read_file(img_path)
img_tf = tf.image.decode_jpeg(img_tf, channels=3)
img_tf = img_tf.numpy()

# read by tensorflow but with another decode
img_tf_2 = tf.io.read_file(img_path)
img_tf_2 = tf.image.decode_jpeg(img_tf_2, channels=3, dct_method='INTEGER_ACCURATE')
img_tf_2 = img_tf_2.numpy()


# read by PIL
img_pil = Image.open(img_path)
img_pil = np.array(img_pil)

img_diff1 = np.abs(img_cv - img_tf)
plt_display(img_diff1, 'OpenCV - TF')


img_diff2 = np.abs(img_cv - img_tf_2)
plt_display(img_diff2, 'OpenCV - TF_accurate_decode')

img_diff3 = np.abs(img_pil - img_cv)
plt_display(img_diff3, 'OpenCV - PIL')

img_diff4 = np.abs(img_pil - img_tf)
plt_display(img_diff4, 'PIL - TF')

plt.show()

