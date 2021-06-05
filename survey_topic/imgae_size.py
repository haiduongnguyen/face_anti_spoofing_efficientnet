"""
first write in 4/6/2021 - do an 2021
this scirpt will count number of image by their size
then show the result by a graph
this will help we understand data for model training
note that the path will only in format:  
    ../train/live/img.jpg
    ../train/spoof/img.jpg

"""

import os 
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.insert(1, '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet')
from config import crop_data_train, crop_data_test


def count_by_size(folder_image_path):
    live_folder_path = folder_image_path + '/live'
    spoof_folder_path = folder_image_path + '/spoof'

    size_dict = {100:0, 200:0, 300:0, 400:0, 500:0, 600:0, 700:0, 800:0}

    for fol in [live_folder_path, spoof_folder_path]:
        for img_name in tqdm(os.listdir(fol)):
            img_path = os.path.join(fol, img_name)
            img = cv2.imread(img_path)
            h, w , _ = img.shape
            s = h + w
            if s < 100:
                size_dict[100] += 1
            elif s < 200:
                size_dict[200] += 1
            elif s < 300:
                size_dict[300] += 1
            elif s < 400:
                size_dict[400] += 1
            elif s < 500:
                size_dict[500] += 1
            elif s < 600:
                size_dict[600] += 1
            elif s < 700:
                size_dict[700] += 1
            else:
                size_dict[800] += 1
    print(size_dict)
    return size_dict


def show_graph(d:dict):
    """
    this function read data from a python dict
    then draw a graph to visualize data in dict
    """
    x = []
    y = []
    for key, value in d.items():
        x.append(str(key))
        y.append(value)

    x_pos = [i for i, _ in enumerate(x)]
    plt.figure()
    plt.bar(x_pos, y, color='green')
    plt.xlabel("Size")
    plt.ylabel("Number of images")
    plt.title("Count by size")
    plt.xticks(x_pos, x)



train_path = crop_data_train
d1 = count_by_size(train_path)
show_graph(d1)

test_path = crop_data_test
d2 = count_by_size(test_path)
show_graph(d2)

plt.show()


# this is the result after run:
# size_train= {100: 1077, 200: 17298, 300: 36400, 400: 42121, 500: 39558, 600: 34338, 700: 19799, 800: 33470}
# size_test = {100: 219, 200: 3483, 300: 6772, 400: 6289, 500: 7343, 600: 5800, 700: 4109, 800: 10378}
#

