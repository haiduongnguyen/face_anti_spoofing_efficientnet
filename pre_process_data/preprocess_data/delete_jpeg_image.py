'''
I see my phone sometimes has take picture and save as .jpeg, so I want to check if there is any image like this
but result show no image with .jpeg tail, so its good.
'''



import glob 
from posix import listdir
import cv2
import numpy as np
import os
from tqdm import tqdm


def find_jpeg_image(path_to_image_folder):
    jpeg_list = glob.glob1(path_to_image_folder, "*.jpeg")
    with open("jpeg_list.txt", 'a') as f:
        for name_image in tqdm(jpeg_list):
            path_image = os.path.join(path_to_image_folder, name_image)
            print('\n' + path_image , file=f) 


def main():
    path_to_sang_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_sang'
    path_to_toi_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_toi'

    for name_folder in tqdm(sorted(os.listdir(path_to_sang_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_sang_folder, name_folder)
        find_jpeg_image(path_to_image_folder)

    for name_folder in tqdm(sorted(os.listdir(path_to_toi_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_toi_folder, name_folder)
        find_jpeg_image(path_to_image_folder)   

    print("complete find all jpeg images, name of them is saved in jpeg_list.txt")



if __name__ == "__main__":
    main()




