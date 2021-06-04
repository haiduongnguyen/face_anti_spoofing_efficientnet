import numpy as np
import os 
from tqdm import tqdm


with open('bad_images.txt') as f:
    lines = f.read().splitlines()

count = 0

with open('duplicate_image.txt', 'w') as f:
    f.close()

f = open('duplicate_image.txt', 'a')
for i in range(len(lines)):
    path = lines[i]
    name = path.split('/')[-1]
    for j in range(i+1, len(lines)):
        if name in lines[j]:
            print(name, file=f)
            count += 1


print(count)
        






