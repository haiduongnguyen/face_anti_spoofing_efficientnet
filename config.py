"""
the file contain all path use in code
just need to change the parameter to run
"""
import os

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

work_place = '/home/duongnh/project_f19/photo_attack/code/face_anti_spoofing_efficientnet'


data_path = '/home/duongnh/project_f19/photo_attack/data'
crop_folder = data_path + '/data_crop'

crop_data_train = crop_folder + '/train'

crop_data_test = crop_folder + '/test'


result_all_model = '/home/duongnh/project_f19/photo_attack/result'


# folder of network to detect face
detector =  work_place + '/face_detector'



original_metas = work_place + '/make_json/metas' 

# json here is original label 43 attribute
folder_save_json =  work_place + '/folder_save_json'
# make_dir(folder_save_json)
helper_train_json_path = folder_save_json + '/helper_train_json.json'
helper_test_json_path =  folder_save_json + '/helper_test_json.json'
