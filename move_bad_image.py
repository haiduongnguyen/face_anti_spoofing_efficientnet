import os
import shutil
from tqdm import tqdm

a_file = open("bad_images.txt", "r")

list_of_paths = []
for line in a_file:
  stripped_line = line.strip()
  list_of_paths.append(stripped_line)

a_file.close()


count = 0
for line in tqdm(list_of_paths):
    # line = line[:-3]
    count += 1
    # print("Line{}: {}".format(count, line.strip()))
    partitions = line.split("/")
    # new_destination = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet/bad_image/' + partitions[-1]
    # shutil.move(line, new_destination)
    print(line)

    
print(count)
print("complete move bad images to new folder")

# 3 ways to move a file to new destination

# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


# an example of a bad image path 

# /home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet/data/crop/train/live/60002474_crop_1.jpeg




