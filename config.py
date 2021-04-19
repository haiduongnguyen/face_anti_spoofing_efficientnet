"""
the file contain all path use in code
just need to change the parameter to run
"""
import keras
import os

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

work_place = '/home/duongnh/face_anti_spoofing_use_resnet50_backbone'

raw_data = '/mnt/DATA/photo/photo_celeb/CelebA_Spoof/Data'

data_train_path = raw_data + '/train'
data_test_path  = raw_data + '/test'


crop_folder = work_place + '/data/crop'
make_dir(crop_folder)
crop_data_train = crop_folder + '/train'
make_dir(crop_data_train)
# crop_data_valid = crop_folder + '/valid'
# make_dir(crop_data_valid)
crop_data_test = crop_folder + '/test'
make_dir(crop_data_test)

original_metas = work_place + '/make_json/metas' 

# json here is original label 43 attribute
folder_save_json =  work_place + '/folder_save_json'
make_dir(folder_save_json)
helper_train_json_path = folder_save_json + '/helper_train_json.json'
helper_test_json_path =  folder_save_json + '/helper_test_json.json'

# json here is only 1 attribute: live or spoof
# not use now
# train_json = folder_save_json +  '/train_label.json'
# test_json = folder_save_json + '/test_label.json'

# folder of network to detect face
detector =  work_place + '/face_detector'

# image parameter
image_size = 224
image_depth = 3

# config parameter to train
INIT_LR = 1e-4
batch_size = 8
EPOCHS = 10

opt_adam = keras.optimizers.Adam(lr=INIT_LR)
opt_sgd = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

# folder save log 
folder_save_log =  work_place + '/log'
make_dir(folder_save_log)

# folder save model after trained
folder_save_model = work_place


#model_name = 'resnet50'
#model_name = 'efficient_net_b7'
model_name = 'efficient_net_b4'
