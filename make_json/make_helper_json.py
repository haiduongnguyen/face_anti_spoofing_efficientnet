# Necessary packages
import os
import json
import sys
sys.path.insert(1, '/home/duongnh/face_anti_spoofing_use_resnet50_backbone')

# My packages
from config import  raw_data, folder_save_json, original_metas
from utils import get_path, extract_sub_images, check_exist_folder


def main():
    # Paths
    train_path = raw_data + '/train'
    test_path = raw_data + '/test'

    imgs_train = get_path(train_path, mode='train')
    print('train set have images:  ', len(imgs_train))
    imgs_test = get_path(test_path, mode='test')
    print('test set have images:  ', len(imgs_test))

    # Metas path
    metas = original_metas
    intra_test_path = metas + '/intra_test'
    #Loading and processing intra_test json
    train_json = json.load(open(intra_test_path + '/train_label.json'))
    print(f"original train json have {len(train_json)} objects")

    test_json = json.load(open(intra_test_path + '/test_label.json'))
    print(f"original test json have {len(test_json)} objects")


    # need to careful of new_destination in extract_sub_image   
    # origin label: folder of author
    # new_destination: my folder that contain images
    train_dict = extract_sub_images(train_json, imgs_train, raw_data)
    print('complete check train images ', len(train_dict))

    test_dict = extract_sub_images(test_json, imgs_test, raw_data)
    print('complete check test image ', len(test_dict))
    

    check_exist_folder(folder_save_json)

    with open(os.path.join(folder_save_json, 'helper_train_json.json'), 'w') as outfile:
        json.dump(train_dict, outfile)
        print('complete dump train json')

    
    with open(os.path.join(folder_save_json, 'helper_test_json.json'), 'w') as outfile:
        json.dump(test_dict, outfile)
        print('complete dump test json')


if __name__ == "__main__":
    main() 
