""""
the file path of wrong sample is at result_model_name/test_flow_from_directory_cp_xx/wrong_sample_path.txt
the path is saved as 
    ...
    live/A3_30000666_crop_3.jpeg
    live/A3_30000674_crop_1.jpeg
    live/A3_30000677_crop_1.jpeg
    ...
this script will read path from file txt and copy image into new folder 
then we will research why model predict wrong these images
"""

import os
from tkinter.constants import W
import numpy as np
from config import work_place
from tqdm import tqdm
import shutil


def make_dir(path):
    if not os.path.isdir(path):
        # os.makedirs(path)
        pass

def copy_wrong_image(model_name, wrong_path_file, base_data):
    base_data = work_place + '/data_crop/test'
    wrong_image_folder_all_model = work_place + '/wrong_image_folder'
    make_dir(wrong_image_folder_all_model)

    wrong_image_folder = wrong_image_folder_all_model + '/' + model_name
    make_dir(wrong_image_folder)
    live_wrong = wrong_image_folder + '/live'
    make_dir(live_wrong)
    spoof_wong = wrong_image_folder + '/spoof'
    make_dir(spoof_wong)

    with open(wrong_path_file) as f:
        path_list = f.read().splitlines()

    for item in tqdm(path_list):
        origin_image = os.path.join(base_data, item)
        new_destination = os.path.join(wrong_image_folder, item)
        shutil.copy2(origin_image, new_destination)
        # print(origin_image)
        # print(new_destination)



model_name = 'new_b0_ver4_cp_01'
wrong_path_file = work_place + '/result_new_b0_ver4/test_flow_from_directory_cp_01/wrong_sample_path.txt'


copy_wrong_image(model_name, wrong_path_file)


