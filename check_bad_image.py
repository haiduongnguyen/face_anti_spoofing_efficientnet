import os
from PIL import Image
from config import work_place
import cv2
from tqdm import tqdm

dir = work_place + '/data/crop'

bad_list=[]

subdir_list = os.listdir(dir)                   # create a list of the sub directories in the directory ie train or test

for d in subdir_list:                           # iterate through the sub directories train and test

    dpath=os.path.join (dir, d)                 # create path to sub directory

    class_list=os.listdir(dpath)                # list of classes ie dog or cat

    for klass in class_list:                    # iterate through the two classes
        class_path=os.path.join(dpath, klass)   # path to class directory

        file_list=os.listdir(class_path)        # create list of files in class directory
        for f in tqdm(file_list):                     # iterate through the files
            fpath=os.path.join (class_path,f)
            index=f.rfind('.')                  
            ext=f[index+1:]                        # get the files extension
            if ext  not in ['jpg', 'png', 'bmp']:
                print(f'file {fpath}  has an invalid extension {ext}')
                bad_list.append(fpath)                    
            else:
                try:
                    im = Image.open(fpath)
                    # img=cv2.imread(fpath)
                    # size=img.shape
                except:
                    bad_list.append(fpath)

textfile = open("bad_images.txt", "w")
for element in bad_list:
    textfile.write(element + "\n")
textfile.close()
