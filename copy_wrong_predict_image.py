import os
import numpy as np
from config import *
from tqdm import tqdm
import shutil

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def copy_wrong_img(model_name, txt_live_wrong, txt_spoof_wrong):
    wrong_image_folder = work_place + '/wrong_image_' + model_name 
    make_dir(wrong_image_folder)
    wrong_live = wrong_image_folder + '/live'
    make_dir(wrong_live)
    wrong_spoof = wrong_image_folder + '/spoof'
    make_dir(wrong_spoof)


    list_of_index = []
    f =  open(txt_live_wrong, 'r') 
    for line in f:
        stripped_line = line.strip()
        list_of_index.append(int(stripped_line))
    f.close()
    
    f =  open(txt_spoof_wrong, 'r') 
    for line in f:
        stripped_line = line.strip()
        list_of_index.append(int(stripped_line))
    f.close()



    path_live = os.path.join(crop_data_test, 'live')
    path_spoof = os.path.join(crop_data_test, 'spoof')

    list_img = []
    count_live = 0
    for img_name in tqdm(os.listdir(path_live)):
        img_path = os.path.join(path_live, img_name)
        list_img.append(img_path)
        count_live += 1

    count_spoof = 0
    for img_name in tqdm(os.listdir(path_spoof)):
        img_path = os.path.join(path_spoof, img_name)
        list_img.append(img_path)  
        count_spoof += 1

    # list_live = os.listdir(path_live)
    # list_spoof = os.listdir(path_spoof)

    # list_all = list_live + list_spoof


    for index, path in enumerate(tqdm(list_img)):
        if not os.path.exists(path):
            print("path is error: ", path)
        if index in list_of_index:
            if index < count_live:
                img_name = path.split("/")[-1]
                new_path = os.path.join(wrong_live, img_name )
                shutil.copy2(path, new_path)
            else:
                img_name = path.split("/")[-1]
                new_path = os.path.join(wrong_spoof, img_name )
                shutil.copy2(path, new_path)            
    print("complete copy wrong image to new folder")


if __name__ == '__main__':

    model_name = 'b1_ver01'

    result_test_folder = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/result_b1_ver01/test_flow_from_directory_cp_06'
    txt_live_wrong =  result_test_folder + '/wrong_live_sample.txt'
    txt_spoof_wrong = result_test_folder + '/wrong_spoof_sample.txt'

    copy_wrong_img(model_name, txt_live_wrong, txt_spoof_wrong)



