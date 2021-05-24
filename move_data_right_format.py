import os 
import shutil
from tqdm import tqdm

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


base_folder = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet'

origin_data = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet/photo_crop'

target_folder = base_folder + '/data_crop'
make_dir(target_folder)

train_folder = target_folder + '/train'
make_dir(train_folder)
test_folder = target_folder + '/test'
make_dir(test_folder)

train_live_folder = train_folder + '/live'
make_dir(train_live_folder)
train_spoof_folder = train_folder + '/spoof'
make_dir(train_spoof_folder)

test_live_folder = test_folder + '/live'
make_dir(test_live_folder)
test_spoof_folder = test_folder + '/spoof'
make_dir(test_spoof_folder)


list_folder = []

def check_folder(path):
    if os.path.isdir(path):
        for tail in os.listdir(path):
            sub_folder = os.path.join(path, tail)
            if os.path.isfile(sub_folder):
                list_folder.append(path)
                break    
                return False
            else:
                return True


folder_path = origin_data

if check_folder(folder_path):
    for sub1 in os.listdir(folder_path):
        sub1_folder = os.path.join(folder_path, sub1)
        if check_folder(sub1_folder):
            for sub2 in os.listdir(sub1_folder):
                sub2_folder = os.path.join(sub1_folder, sub2)
                if check_folder(sub2_folder):
                    for sub3 in os.listdir(sub2_folder):
                        sub3_folder = os.path.join(sub2_folder, sub3)
                        if check_folder(sub3_folder):
                            for sub4 in os.listdir(sub3_folder):
                                sub4_folder = os.path.join(sub3_folder, sub4)
                                if check_folder(sub4_folder):
                                    for sub5 in os.listdir(sub4_folder):
                                        sub5_folder = os.path.join(sub4_folder, sub5)
                                        if check_folder(sub5_folder):
                                            for sub6 in os.listdir(sub5_folder):
                                                sub6_folder = os.path.join(sub6_folder, sub6)
                                            if check_folder(sub6_folder):
                                                print("hoi bi nhieu thu muc qua r day !!!")                                  


print(len(list_folder))


def copy_dir_to_dir(source, destination):
    for file_name in os.listdir(source):
        s = os.path.join(source, file_name)
        d = os.path.join(destination, file_name)
        shutil.copy2(s, d)

for folder_path in tqdm(list_folder):
    if 'train' in folder_path:
        if '/live/' in folder_path:
            copy_dir_to_dir(folder_path, train_live_folder)
        elif '/spoof/' in folder_path:
            copy_dir_to_dir(folder_path, train_spoof_folder)
        else:
            print("the path is error format: " + folder_path )

    elif 'test' in folder_path:
        if '/live/' in folder_path :
            copy_dir_to_dir(folder_path, test_live_folder)
        elif '/spoof/' in folder_path:
            copy_dir_to_dir(folder_path, test_spoof_folder)
        else:
            print("the path is error format: " + folder_path )
    else:
        print("the path is error format: " + folder_path )

