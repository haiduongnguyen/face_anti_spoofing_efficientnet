"""
the file contain all path use in code
just need to change the parameter to run
"""
import os

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

work_place = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet'

raw_data = ''

data_raw_train_path = raw_data + '/train'
data_raw_test_path  = raw_data + '/test'


crop_folder = work_place + '/data_crop'
# make_dir(crop_folder)
crop_data_train = crop_folder + '/train'
# make_dir(crop_data_train)
# crop_data_valid = crop_folder + '/valid'
# make_dir(crop_data_valid)
crop_data_test = crop_folder + '/test'
# make_dir(crop_data_test)
# for mode in ['train', 'test']:
#     make_dir(os.path.join(crop_folder, mode, 'live'))
#     make_dir(os.path.join(crop_folder, mode, 'spoof'))


original_metas = work_place + '/make_json/metas' 

# json here is original label 43 attribute
folder_save_json =  work_place + '/folder_save_json'
# make_dir(folder_save_json)
helper_train_json_path = folder_save_json + '/helper_train_json.json'
helper_test_json_path =  folder_save_json + '/helper_test_json.json'

# json here is only 1 attribute: live or spoof
# not use now
# train_json = folder_save_json +  '/train_label.json'
# test_json = folder_save_json + '/test_label.json'

# folder of network to detect face
detector =  work_place + '/face_detector'

