'''
my phone dont have enough space, so I divided data (6000 img) into 24 folder
finally, I have to union them to just 1 folder, and maybe change file name in case joint with other data
the data after union is in  Desktop/6_xiaomi
'''


import shutil
import glob 
from posix import listdir
import cv2
import numpy as np
import os
from tqdm import tqdm


def copy_image_from_folder(path_to_image_folder, path_to_union_folder):
    partition = path_to_image_folder.split('/')
    
    for name_image in tqdm(os.listdir(path_to_image_folder)):

        path_to_raw_iamge = os.path.join(path_to_image_folder, name_image)

        path_to_new_image = os.path.join(path_to_union_folder, partition[-1] + name_image[3:])

        shutil.copy2(path_to_raw_iamge, path_to_new_image)



def main():
    path_to_sang_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_sang'
    path_to_toi_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_toi'
    
    path_to_union_folder_sang = '/home/duong/Desktop/6_xiaomi/sang'
    path_to_union_folder_toi = '/home/duong/Desktop/6_xiaomi/toi'


    for name_folder in tqdm(sorted(os.listdir(path_to_sang_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_sang_folder, name_folder)
        copy_image_from_folder(path_to_image_folder, path_to_union_folder_sang)

    for name_folder in tqdm(sorted(os.listdir(path_to_toi_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_toi_folder, name_folder)
        copy_image_from_folder(path_to_image_folder, path_to_union_folder_toi)   

    print("complete copy all image to union folder, result is at Desktop/6_xiaomi")



if __name__ == "__main__":
    main()




