'''
Sometimes my phone take photo and save them with _HHT_ in file name
I want to see them if they have error or something else **just be careful**
the list of them will be save in HHT_list.txt in preprocess_data
some useful information:
    HDR:
    HDR is for High Dynamic Range. When the scene has very deep shadows and very bright highlights HDR can balance it. 
    It can be made from one image or several images (taken at different exposure values) combined into one.

    HHT:
    The HHT (Handheld Twilight) mode takes multiple sequential images under low light conditions, 
    aligns them, and then create a single optimized image. It allows you to take a sharper low-light picture without use of tripod
'''


import glob 
from posix import listdir
import cv2
import numpy as np
import os
from tqdm import tqdm

with open('HHT_list.txt', 'w') as f:
    f.close()

def find_HHT_image(path_to_image_folder):
    for name_image in tqdm(os.listdir(path_to_image_folder)):
        if 'HHT' in name_image:
            path_image = os.path.join(path_to_image_folder, name_image)
            with open('HHT_list.txt', 'a') as f:
                print(path_image, file=f)

def main():
    path_to_sang_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_sang'
    path_to_toi_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_toi'

    for name_folder in tqdm(sorted(os.listdir(path_to_sang_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_sang_folder, name_folder)
        find_HHT_image(path_to_image_folder)

    for name_folder in tqdm(sorted(os.listdir(path_to_toi_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_toi_folder, name_folder)
        find_HHT_image(path_to_image_folder)   

    print("complete find all jpeg images, name of them is saved in jpeg_list.txt")



if __name__ == "__main__":
    main()




