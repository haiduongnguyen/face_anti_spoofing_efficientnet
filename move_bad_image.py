import os
import shutil
from tqdm import tqdm

file1 = open('bad_images.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
for line in tqdm(Lines):
    count += 1
    # print("Line{}: {}".format(count, line.strip()))
    partitions = line.split("/")
    new_destination = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet/bad_image/' + partitions[-1]
    shutil.move(line, new_destination)

print(count)
print("complete move bad images to new folder")


# 3 ways to move a file to new destination

# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


# an example of a bad image path 

# /home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet/data/crop/train/live/60002474_crop_1.jpeg




