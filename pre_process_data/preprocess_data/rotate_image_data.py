from posix import listdir
import cv2
import numpy as np
import os
from tqdm import tqdm

from PIL import Image


width = 3120
height = 4160

# define a function that rotates images in the current directory
# given the rotation in degrees as a parameter

def rotate_an_image(path_to_img, angle):
    # rotate and save the image with the same filename
    img = Image.open(path_to_img)

    img = img.rotate(angle)

    # resize to to right size
    img.resize((width,height)).save(path_to_img)
    # close the image
    img.close()



def rotate_all_folder(path_to_image_folder):
    # for each image in the current directory
    for index, name_image in tqdm(enumerate(sorted(os.listdir(path_to_image_folder)))):
        # open the image
        path_to_img = os.path.join(path_to_image_folder, name_image)

        img = Image.open(path_to_img)
        w, h = img.size

        if index % 5 == 0:
            if w > h:
                rotate_an_image(path_to_img, 270)

        elif index % 5 == 1:
            rotate_an_image(path_to_img, 180)

        elif index % 5 == 2:
            if w > h:
                rotate_an_image(path_to_img, 90)

        elif index % 5 == 3:
            pass

        elif index % 5 == 4:
            if w > h:
                rotate_an_image(path_to_img, 270)        
        index += 1
    

def main():
    path_to_sang_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_sang'
    path_to_toi_folder = '/home/duong/Desktop/6_hduong_redmi_note_4X/tep_6_hduong_xiaomi_toi'

    for name_folder in tqdm(sorted(os.listdir(path_to_sang_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_sang_folder, name_folder)
        rotate_all_folder(path_to_image_folder)

    for name_folder in tqdm(sorted(os.listdir(path_to_toi_folder))):
        # print(name_folder)
        path_to_image_folder = os.path.join(path_to_toi_folder, name_folder)
        rotate_all_folder(path_to_image_folder)   

    print("complete rotate all images")



if __name__ == "__main__":
    main()

