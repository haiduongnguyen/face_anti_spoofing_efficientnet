import os
from PIL import Image

import cv2
from tqdm import tqdm

dir = '/home/duong/project/pyimage_research/version2_change_data/small_bad_images'

bad_list=[]
count = 0

for f in tqdm(os.listdir(dir)):                     # iterate through the files
    fpath=os.path.join (dir,f)
    index=f.rfind('.')                  
    ext=f[index+1:]                        # get the files extension
    if ext  not in ['jpg', 'png', 'bmp', 'jpeg']:
        print(f'file {fpath}  has an invalid extension {ext}')
        bad_list.append(fpath)                    
    else:
        try:
            # im =cv2.imread(fpath)
            im = Image.open(fpath)
            # img=cv2.imread(fpath)
            # size=img.shape
        except:
            count += 1
            bad_list.append(fpath)

for item in bad_list:
    print(item) 


print(count)

# textfile = open("bad_images.txt", "w")
# for element in bad_list:
#     textfile.write(element + "\n")
# textfile.close()
